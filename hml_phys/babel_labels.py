"""BABEL frame-label helpers for the G1 line.

Recovered from the deleted ADAPT reproduction (`adapt/data.py`, commit e67e4ae) and kept here because
our own G1 code depends on them: `scripts/hml_phys/g1_prompt_pool.py`, `scripts/build_text_dict.py` and
`scripts/record_tracker_rollouts.py` all used `clean_label` / `labels_overlapping` to turn BABEL frame
annotations into the text conditioning for the end-to-end G1 policy. They are plain label bookkeeping --
nothing about ADAPT's method survives in them -- so they move here rather than being deleted with the
rest (CLAUDE.md §5, STATUS.md §4).
"""

FPS = 50


def clean_label(text: str) -> str:
    text = text.replace("transition to ", "")
    if "walk back " in text or text == "walk back":
        text = "walk"
    return text


def labels_overlapping(frame_ann, t_start: float, t_end: float, strict: float = 3.0 / FPS,
                       drop=("transition",)):
    """BABEL frame labels overlapping [t_start, t_end], minus `drop`, with `strict` slack at both ends."""
    out = []
    for seg in frame_ann:
        if len(seg) < 3:
            continue
        s, e, label = float(seg[0]), float(seg[1]), str(seg[2])
        if label in drop:
            continue
        if not (s + strict > t_end or t_start > e - strict):
            out.append(clean_label(label))
    return out
