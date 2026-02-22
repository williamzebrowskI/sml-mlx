#!/usr/bin/env python3
"""Pull the latest git changes across all cluster hosts via SSH.

Example:
  python /Users/williamzebrowski/sml-mlx/pull_all_hosts.py
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_HOSTS = ["mac-1.local", "mac-2.local", "mac-3.local", "mac-4.local"]


def run_local(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def detect_branch(repo: Path) -> str:
    branch = run_local(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"])
    if branch == "HEAD":
        raise RuntimeError(
            "Repository is in detached HEAD state. Pass --branch explicitly."
        )
    return branch


def remote_pull_cmd(repo: str, branch: str) -> str:
    # Use sh-compatible command string for broad remote shell compatibility.
    return (
        "set -e; "
        f"cd {shlex.quote(repo)}; "
        f"git fetch origin {shlex.quote(branch)}; "
        f"git checkout {shlex.quote(branch)}; "
        f"git pull --ff-only origin {shlex.quote(branch)}; "
        "printf 'branch=%s head=%s\\n' "
        '"$(git rev-parse --abbrev-ref HEAD)" '
        '"$(git rev-parse --short HEAD)"'
    )


def run_on_host(host: str, repo: str, branch: str, ssh_opts: list[str], dry_run: bool) -> tuple[bool, str]:
    cmd = ["ssh", *ssh_opts, host, remote_pull_cmd(repo, branch)]
    if dry_run:
        return True, shlex.join(cmd)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SSH into each host and run git pull for a repo/branch."
    )
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parent),
        help="Absolute path to repository on all hosts.",
    )
    parser.add_argument(
        "--branch",
        default="",
        help="Branch to pull. Default: auto-detect from local repo.",
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=DEFAULT_HOSTS,
        help="Hosts to update (space-separated).",
    )
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra ssh option token (repeatable), e.g. --ssh-option=-o --ssh-option=ConnectTimeout=5",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue updating other hosts if one host fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ssh commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(f"[error] Repo path does not exist: {repo}", file=sys.stderr)
        return 2

    try:
        branch = args.branch or detect_branch(repo)
    except Exception as exc:
        print(f"[error] Could not determine branch: {exc}", file=sys.stderr)
        return 2

    print(f"[info] repo={repo}")
    print(f"[info] branch={branch}")
    print(f"[info] hosts={', '.join(args.hosts)}")

    failures: list[str] = []
    for host in args.hosts:
        print(f"\n===== {host} =====")
        ok, output = run_on_host(
            host=host,
            repo=str(repo),
            branch=branch,
            ssh_opts=args.ssh_option,
            dry_run=args.dry_run,
        )
        if output:
            print(output)
        if not ok:
            failures.append(host)
            print(f"[error] {host} failed", file=sys.stderr)
            if not args.continue_on_error:
                break
        else:
            print(f"[ok] {host}")

    print("\n===== summary =====")
    if failures:
        print(f"failed_hosts={', '.join(failures)}")
        return 1
    print("all_hosts_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

