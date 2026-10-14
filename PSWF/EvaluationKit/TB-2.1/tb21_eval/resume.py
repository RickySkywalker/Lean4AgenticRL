"""Continue an interrupted harbor job (machine reboot, harbor killed, SSH session lost).

    python -m tb21_eval.resume --run-dir R [--job-name NAME] [--dry-run]

`harbor job resume` keeps every finished trial and re-runs the unfinished ones, but it has no --env-file option,
and a dead harbor process leaves its task containers running. This command therefore

  1. refuses to run while a harbor process of the job is still alive,
  2. removes the leftover containers of the job's unfinished trials (only those; on the guarded network),
  3. runs `harbor job resume --job-path R/jobs/<job>` with R/harbor.env exported (Q38_TOKENIZER_JSON etc.).

Bring the network guard and the serving stack back first if the machine rebooted (`netguard up`, `serve up`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import tb21_eval as T


def die(msg: str) -> None:
    raise SystemExit(f"[resume] FATAL: {msg}")


def load_env_file(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def harbor_alive_for(job_dir: Path, run_dir: Path) -> list[int]:
    """PIDs of running harbor processes whose command line names this job or its run dir's job config."""
    pids = []
    needles = (str(job_dir), str(run_dir / "job.json"), f"{run_dir}/job.")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        words = [a.decode(errors="replace") for a in argv if a]
        if any(w.endswith("harbor") for w in words[:2]) and any(n in w for w in words for n in needles):
            pids.append(int(proc.name))
    return pids


def leftover_containers(job_dir: Path) -> list[str]:
    """Containers on the guarded network that belong to an unfinished trial of this job."""
    unfinished = {t.name.lower() for t in job_dir.iterdir()
                  if t.is_dir() and "__" in t.name and not (t / "result.json").exists()}
    if not unfinished:
        return []
    out = subprocess.run(["docker", "ps", "-a", "--filter", f"network={T.GUARD_NETWORK}", "--format", "{{.Names}}"],
                         capture_output=True, text=True, check=True).stdout.split()
    return [name for name in out if any(name.startswith(f"{trial}__") for trial in unfinished)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="the make_job run dir (holds harbor.env, jobs/)")
    ap.add_argument("--job-name", default=None, help="job to resume (default: the job of R/job.json)")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be done")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    job_name = args.job_name or json.loads((run_dir / "job.json").read_text())["job_name"]
    job_dir = run_dir / "jobs" / job_name
    env_file = run_dir / "harbor.env"
    if not job_dir.is_dir() or not env_file.is_file():
        die(f"{job_dir} or {env_file} does not exist")
    alive = harbor_alive_for(job_dir, run_dir)
    if alive:
        die(f"harbor is still running this job (pid {', '.join(map(str, alive))}); nothing to resume")

    leftovers = leftover_containers(job_dir)
    harbor = str(Path(sys.executable).parent / "harbor")
    cmd = [harbor, "job", "resume", "--job-path", str(job_dir)]
    print(f"[resume] {len(leftovers)} leftover container(s) of unfinished trials: {' '.join(leftovers) or '-'}")
    print(f"[resume] (set -a; . {env_file}; set +a; {' '.join(cmd)})")
    if args.dry_run:
        return 0
    if leftovers:
        subprocess.run(["docker", "rm", "-f", *leftovers], check=True, capture_output=True)
    env = dict(os.environ, **load_env_file(env_file))
    os.execve(harbor, cmd, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
