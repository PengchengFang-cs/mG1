"""A text <-> G1-motion retrieval model, trained on our own robot trajectories.

WHY THIS EXISTS. The question "does the robot do what the caption says" needs a text-motion retrieval
model, and the standard one (Guo et al. 2022, which `hml_phys/evaluator.py` wraps) is trained on SMPL
humans in HumanML3D's 263-d representation. Scoring G1 motion with it requires mapping the robot onto
22 SMPL joints, and measurement showed that path has no discriminative power where we need it: with the
ground truth restricted to the same clips, the retargeted reference scores R@1 0.308 but every EXECUTED
rollout -- the teacher included, and it is handed the correct motion -- lands at 0.048 to 0.081 against
a chance level of 0.031, with the ordering flipping between ground-truth sets (STATUS.md §5.11d). The
evaluator saturates near chance on robot motion.

That is exactly why the field does not do it that way. Of eleven text-to-humanoid papers surveyed, five
train their own text-motion retrieval model on robot trajectories and report R-precision, MM-Dist and
Diversity in that space -- SENTINEL "train[s] a text-motion retrieval (TMR) model on D_robot", DAJI's
"evaluator is trained on paired text descriptions and robot motion clips", LangWBC and Humanoid-LLA
compute retrieval directly in humanoid motion space, TEXEDO uses a learned co-embedding as a selection
objective. None maps robot motion back to SMPL. None runs a user study or a VLM judge. (STATUS.md
§5.11c.)

WHAT IS AND IS NOT BOUGHT BY THIS. Numbers from this model are not comparable to any other paper's --
but neither are theirs to each other, since each trains its own, so this subfield compares R@1 only
within a paper against its own baselines. CLAUDE.md §2's discipline therefore stands unchanged: these
numbers go next to OUR anchors (reference, teacher, behaviour cloning, residual) and never into a table
beside someone else's R@1. Cross-paper claims stay with the physical metrics.

DESIGN. Deliberately the same architecture and the same metric definitions as the Guo evaluator, so
only the motion space changes:

    text    GloVe 300-d word embeddings + 15-d POS one-hot -> TextEncoderBiGRUCo(512) -> 512-d
    motion  G1 proprio, 20 Hz -> MovementConvEncoder(512) -> MotionEncoderBiGRUCo(1024) -> 512-d

The motion side is the robot's OWN state, with no mapping of any kind:

    base_lin_vel 3 | base_ang_vel 3 | projected_gravity 3 | dof_pos 21 | dof_vel 21   = 51

which is what `g1e2e_record_rollouts.py` stored for all 146,200 training trajectories. `projected_gravity`
carries the root's pitch and roll, so orientation is present; absolute root height and heading are not,
the same omission HumanML3D's own representation makes for heading. The ACTION is deliberately excluded:
it is the policy's command rather than the robot's realised motion, and including it would let the
retrieval model key on which policy produced a rollout instead of on what the robot did.

Retrieval is Euclidean, because `hml_phys/t2m/metrics.py:27` scores R-precision with
`euclidean_distance_matrix`; the training loss is InfoNCE on negative squared Euclidean distance so that
the objective and the metric agree.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hml_phys.t2m.t2m_eval_modules import (MotionEncoderBiGRUCo, MovementConvEncoder,
                                           TextEncoderBiGRUCo)

PROPRIO_DIM = 51
SRC_HZ = 50.0           # the rollouts' control rate
TMR_FPS = 20            # Guo's rate, kept so MIN/MAX/UNIT length semantics carry over verbatim
MAX_LEN = 196
MIN_LEN = 40
UNIT_LEN = 4
DIM_WORD = 300
DIM_POS = 15
DIM_COEMB = 512


def resample(x, src_fps=SRC_HZ, dst_fps=TMR_FPS):
    """[T,D] -> [T',D] by linear interpolation, the same scheme hml_phys/sim2hml.py uses."""
    x = np.asarray(x, dtype=np.float32)
    t_src = np.arange(x.shape[0], dtype=np.float64) / src_fps
    n_dst = max(int(round(x.shape[0] / src_fps * dst_fps)), 1)
    t_dst = np.arange(n_dst, dtype=np.float64) / dst_fps
    out = np.empty((n_dst, x.shape[1]), dtype=np.float32)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, x[:, d])
    return out


def motion_feature(proprio, src_fps=SRC_HZ):
    """A rollout's proprio at its control rate -> the TMR's motion feature at 20 Hz. No mapping."""
    p = np.asarray(proprio, dtype=np.float32)
    assert p.ndim == 2 and p.shape[1] == PROPRIO_DIM, p.shape
    return resample(p, src_fps, TMR_FPS)


class G1TMR(nn.Module):
    """Guo's evaluator architecture with the motion branch's input width changed to G1's proprio."""

    def __init__(self, input_dim=PROPRIO_DIM, mov_hidden=512, mov_latent=512,
                 motion_hidden=1024, text_hidden=512, coemb=DIM_COEMB, device="cpu"):
        super().__init__()
        self.input_dim = int(input_dim)
        # Guo passes `dim_pose - 4` here because HumanML3D's last 4 channels are binary foot contacts
        # and it drops them. Our 51 channels are all continuous robot state, so nothing is dropped.
        self.movement = MovementConvEncoder(self.input_dim, mov_hidden, mov_latent)
        self.motion = MotionEncoderBiGRUCo(mov_latent, motion_hidden, coemb, device)
        self.text = TextEncoderBiGRUCo(DIM_WORD, DIM_POS, text_hidden, coemb, device)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def encode_motion(self, feat, m_lens):
        """feat [B,T,51] normalised, m_lens in 20 Hz frames -> [B,512]. m_lens must be DESCENDING."""
        mov = self.movement(feat)
        # two stride-2 convolutions, so the movement sequence is a quarter as long
        return self.motion(mov, torch.div(m_lens, UNIT_LEN, rounding_mode="floor"))

    def encode_text(self, word_embs, pos_onehot, cap_lens):
        """cap_lens must be DESCENDING -- both branches use packed GRUs."""
        return self.text(word_embs, pos_onehot, cap_lens)

    def co_embed(self, word_embs, pos_onehot, cap_lens, motions, m_lens):
        """Aligned (text, motion) embeddings, exactly as t2m_eval_wrapper.get_co_embeddings does it.

        Both branches are packed GRUs, so each needs its own input sorted by length descending. The
        caller sorts the batch by CAPTION length; the motion side is re-sorted by MOTION length here
        and the text embedding is permuted by the same index, so row i of both outputs is still the
        same pair. Getting this wrong silently pairs each caption with another clip's motion.
        """
        align = torch.argsort(m_lens, descending=True)
        me = self.encode_motion(motions[align], m_lens[align])
        te = self.encode_text(word_embs, pos_onehot, cap_lens)[align]
        return te, me

    def loss(self, te, me):
        """Symmetric InfoNCE on negative squared Euclidean distance -- the metric's own geometry."""
        d2 = torch.cdist(te, me).pow(2)
        logits = -d2 * self.log_temp.exp()
        tgt = torch.arange(te.shape[0], device=te.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def save_tmr(path, model, mean, std, meta):
    torch.save(dict(model=model.state_dict(), mean=np.asarray(mean), std=np.asarray(std),
                    input_dim=model.input_dim, meta=meta), path)


def load_tmr(path, device="cpu"):
    ck = torch.load(path, map_location="cpu")
    m = G1TMR(input_dim=int(ck["input_dim"]), device=device).to(device)
    m.load_state_dict(ck["model"])
    m.eval().requires_grad_(False)
    return m, np.asarray(ck["mean"], np.float32), np.asarray(ck["std"], np.float32), ck.get("meta", {})
