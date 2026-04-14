#!/usr/bin/env python
"""Download the published checkpoint bundle into $DIFFSCI2_MODEL_ROOT.

Expected layout after a successful run:

    $DIFFSCI2_MODEL_ROOT/
    ├── pore/
    │   ├── production/
    │   │   ├── converted_vaenet.ckpt
    │   │   ├── bentheimer_pcond.ckpt
    │   │   ├── doddington_pcond.ckpt
    │   │   ├── estaillades_pcond.ckpt
    │   │   └── ketton_pcond.ckpt
    │   └── field_controlled/
    │       ├── fc257/{bentheimer,doddington,estaillades,ketton}.ckpt
    │       └── fc129/{bentheimer,doddington,estaillades,ketton}.ckpt
    └── (experimental/ is untouched — dated runs are authoritative there)

The bundle is distributed on Zenodo (see docs/checkpoints.md for the DOI).
This script reads a `MANIFEST.json` from the Zenodo record, downloads each
file listed, verifies its sha256, and places it at the canonical path.

Before the first Zenodo upload happens (see "User action list" in
claude/report/CODE_ORGANIZATION.md), the manifest URL below is a
placeholder. Editing MANIFEST_URL to point at the real record is a
one-line change; the rest is already in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from pipelines._common import model_root  # noqa: E402


# ---------------------------------------------------------------------------
# Placeholder — replace with the real Zenodo record URL once uploaded.
# ---------------------------------------------------------------------------
MANIFEST_URL = os.environ.get(
    "DIFFSCI2_CHECKPOINT_MANIFEST_URL",
    "https://zenodo.org/records/PLACEHOLDER/files/MANIFEST.json",
)


CHUNK = 1 << 20  # 1 MB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha and sha256_of(dest) == expected_sha:
        print(f"  [skip] {dest.relative_to(model_root())}  (sha256 match)")
        return
    print(f"  [get ] {dest.relative_to(model_root())}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    if expected_sha:
        got = sha256_of(dest)
        if got != expected_sha:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {dest}: expected {expected_sha}, got {got}"
            )


SET_TO_TAGS = {
    "base":   {"vae", "base"},
    "fc":     {"fc257", "fc129"},
    "uncond": {"uncond"},
    "all":    {"vae", "base", "fc257", "fc129", "uncond"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--set", default="all",
                   choices=sorted(SET_TO_TAGS.keys()),
                   help="Which subset of the bundle to fetch (default: all).")
    p.add_argument("--manifest-url", default=MANIFEST_URL,
                   help=f"Manifest URL (default: {MANIFEST_URL}).")
    p.add_argument("--manifest-file", default=None,
                   help="Local MANIFEST.json path (overrides --manifest-url).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not HAVE_REQUESTS and args.manifest_file is None:
        raise SystemExit("`requests` package required; install with `pip install requests`.")

    mr = model_root()
    mr.mkdir(parents=True, exist_ok=True)

    if args.manifest_file:
        manifest = json.loads(Path(args.manifest_file).read_text())
    else:
        print(f"Fetching manifest: {args.manifest_url}")
        try:
            resp = requests.get(args.manifest_url, timeout=30)
            resp.raise_for_status()
            manifest = resp.json()
        except Exception as e:
            raise SystemExit(
                f"Could not fetch manifest from {args.manifest_url}: {e}\n\n"
                "Hint: the manifest URL is a placeholder until the Zenodo "
                "upload is done. See docs/checkpoints.md and §U.1 of "
                "claude/report/CODE_ORGANIZATION.md. "
                "You can also run this script with --manifest-file path/to/MANIFEST.json."
            )

    wanted_tags = SET_TO_TAGS[args.set]
    entries = [e for e in manifest.get("files", []) if e.get("tag", "") in wanted_tags]
    print(f"\n{len(entries)} file(s) to consider for set={args.set}")

    for entry in entries:
        target = mr / entry["target_path"]
        download(entry["url"], target, entry.get("sha256"))

    print("\nDone.")


if __name__ == "__main__":
    main()
