
#!/usr/bin/env python3
"""Download Python slice of The Stack v2 and optionally StarCoderData extras.

Example:
    python fetch_stack.py --output data/raw
"""
import argparse, os, subprocess, pathlib, json, textwrap, sys

STACK_URL = "https://huggingface.co/datasets/bigcode/the-stack-dedup-v2"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="output directory")
    args = ap.parse_args()

    out = pathlib.Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print("▶️  Downloading The Stack…")
    cmd = [
        "huggingface-cli", "download",
        STACK_URL,
        "--include", "python"
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=out)

    print("✅  done. Files saved under", out)

if __name__ == "__main__":
    main()
