"""Launch a long run that outlives the shell that started it.

macOS has no `setsid`, and `nohup` alone is not enough: it only ignores SIGHUP, so a job still
dies when its parent's whole process group is signalled. `start_new_session=True` gives the
child its own session and process group, which is what actually detaches it. `caffeinate -is`
keeps the Mac awake for the duration, since these runs are hours long.

    .venv/bin/python utils/setup/detach.py logs/grid.log -- .venv/bin/python -u -m src.train --K 5
"""
import argparse
import subprocess
from pathlib import Path


def launch(log_path: str, cmd: list[str]) -> int:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log, "w")          # inherited by the child; it closes when the child exits
    proc = subprocess.Popen(["caffeinate", "-is", *cmd], stdout=handle,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    return proc.pid


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="file that receives stdout and stderr")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run, after --")
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("no command given")
    print(f"pid {launch(a.log, cmd)}, logging to {a.log}")
