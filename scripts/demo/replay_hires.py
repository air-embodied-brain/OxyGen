"""Replay a logged LIBERO rollout at high resolution and verify against logged oracle states.

Run in the LIBERO client environment, e.g.
  PYTHONPATH=<openpi_subtask_generation>/third_party/libero MUJOCO_GL=egl \
    <client_venv>/bin/python replay_hires.py --run <run_dir> --task 6 --episode 3 --out <dir>
"""
import argparse, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
DUMMY = [0.0] * 6 + [-1.0]

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--task", type=int, required=True)
p.add_argument("--episode", type=int, required=True); p.add_argument("--res", type=int, default=768)
p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(a.task)
env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                         camera_heights=a.res, camera_widths=a.res)
env.seed(a.seed); env.reset()
obs = env.set_init_state(suite.get_task_init_states(a.task)[a.episode])
goals = [list(s) for s in env.env.parsed_problem["goal_state"]]
acts = [json.loads(l) for l in open(Path(a.run) / "action_trajectory.jsonl")]
acts = [x for x in acts if x["episode_idx"] == a.episode]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
frames = {}
for _ in range(10):
    obs, *_ = env.step(DUMMY)
mism = 0
for x in acts:
    vals = [bool(env.env._eval_predicate(s)) for s in goals]
    if vals != x["oracle_before_action"]["goal_values"]:
        mism += 1
    frames[x["step"]] = obs["agentview_image"][::-1, ::-1].copy()
    obs, r, done, _ = env.step(x["action"])
frames[acts[-1]["step"] + 1] = obs["agentview_image"][::-1, ::-1].copy()
steps = sorted(frames)
np.savez_compressed(out / f"t{a.task}_e{a.episode}.npz", steps=np.array(steps), frames=np.stack([frames[s] for s in steps]))
print(json.dumps({"steps": len(acts), "oracle_mismatches": mism, "done": bool(done)}))
