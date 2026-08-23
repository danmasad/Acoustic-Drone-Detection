"""Launch a long run that outlives the shell that started it.

macOS has no `setsid`, and `nohup` alone is not enough: it only ignores SIGHUP, so a job still
dies when its parent's whole process group is signalled. `start_new_session=True` gives the
child its own session and process group, which is what actually detaches it. `caffeinate -is`
keeps the Mac awake for the duration, since these runs are hours long.

    .venv/bin/python utils/detach.py logs/grid.log -- .venv/bin/python -u -m src.train --K 5
"""
import subprocess
import sys
from pathlib import Path


def launch(log_path: str, cmd: list[str]) -> int:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log, "w")
    proc = subprocess.Popen(["caffeinate", "-is", *cmd], stdout=handle,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    return proc.pid


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    log_path, cmd = args[0], args[1:]
    if cmd[0] == "--":
        cmd = cmd[1:]
    pid = launch(log_path, cmd)
    print(f"pid {pid}, logging to {log_path}")
