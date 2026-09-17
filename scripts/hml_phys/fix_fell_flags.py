"""One-off repair for rollouts recorded before the terminate-buffer fix.

The env `done` used for fall bookkeeping also fires at the episode timeout (progress_buf >= episode_length,
which the 2 warm-up steps make happen at executed step episode_length-2). Episodes whose recorded fall_step
equals that timeout step are NOT falls. The timeout step is taken as the maximum fall_step in the file.
"""
import sys, collections, joblib
p = sys.argv[1]
d = joblib.load(p)
steps = collections.Counter(e["fall_step"] for e in d["episodes"] if e["fall_step"] != -1)
timeout_step = max(steps) if steps else None
print("fall_step histogram (top 5):", steps.most_common(5), "-> timeout step", timeout_step)
n0 = sum(e["fell"] for e in d["episodes"])
for e in d["episodes"]:
    real = e["fall_step"] != -1 and e["fall_step"] < timeout_step and e["fall_step"] < e["target_frames"]
    e["fell"] = bool(real)
    if not real:
        e["fall_step"] = -1
n1 = sum(e["fell"] for e in d["episodes"])
d["meta"]["fell_flags_repaired"] = True; d["meta"]["timeout_step"] = timeout_step
joblib.dump(d, p)
print(f"{p}: fell {n0} -> {n1} of {len(d['episodes'])}")
