#!/usr/bin/env python
"""Download the Imperial College four-rock micro-CT volumes.

Populates $DIFFSCI2_DATA_ROOT (default ./saveddata/raw/imperial_college)
with the .raw + .mhd + .network.npz triples referenced by the rock YAML
configs under pipelines/field_controlled/configs/.

Source:
    https://www.imperial.ac.uk/earth-science/research/research-groups/
    pore-scale-modelling/micro-ct-images-and-networks/

Because the Imperial College page does not provide direct stable URLs
for the individual rock archives, this script reads a small JSON
manifest (default: tools/data_manifest.json) that lists per-rock URLs
and sha256 hashes. If no manifest is present, the script prints the
rocks that are still missing and the list of canonical filenames
expected on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from pipelines._common import data_root  # noqa: E402


CANONICAL_FILES = [
    "Bentheimer_1000c_3p0035um.raw",
    "Bentheimer_1000c_3p0035um.mhd",
    "Bentheimer_1000c_3p0035um.network.npz",
    "Doddington_1000c_2p6929um.raw",
    "Doddington_1000c_2p6929um.mhd",
    "Doddington_1000c_2p6929um.network.npz",
    "Estaillades_1000c_3p31136um.raw",
    "Estaillades_1000c_3p31136um.mhd",
    "Estaillades_1000c_3p31136um.network.npz",
    "Ketton_1000c_3p00006um.raw",
    "Ketton_1000c_3p00006um.mhd",
    "Ketton_1000c_3p00006um.network.npz",
]

DEFAULT_MANIFEST = REPO_ROOT / "tools" / "data_manifest.json"


CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(CHUNK), b""):
            h.update(c)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha and sha256_of(dest) == expected_sha:
        print(f"  [skip] {dest.name} (sha256 match)")
        return
    print(f"  [get ] {dest.name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk_size=CHUNK):
                if c:
                    f.write(c)
        tmp.replace(dest)
    if expected_sha:
        got = sha256_of(dest)
        if got != expected_sha:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"sha256 mismatch for {dest}: expected {expected_sha}, got {got}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                   help="Path to a JSON manifest listing URLs + sha256s per "
                        "canonical filename. Default: tools/data_manifest.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dr = data_root()
    dr.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        missing = [f for f in CANONICAL_FILES if not (dr / f).exists()]
        print(f"No manifest found at {manifest_path}.")
        print(f"Data root: {dr}")
        print(f"Missing files ({len(missing)} of {len(CANONICAL_FILES)}):")
        for f in missing:
            print(f"  {f}")
        print()
        print("Obtain the four-rock bundle from the Imperial College page:")
        print("  https://www.imperial.ac.uk/earth-science/research/")
        print("  research-groups/pore-scale-modelling/micro-ct-images-and-networks/")
        print()
        print("Place the files directly under the data root, OR create a manifest")
        print("at tools/data_manifest.json with entries like:")
        print(json.dumps({
            "files": [
                {
                    "filename": "Bentheimer_1000c_3p0035um.raw",
                    "url": "https://.../Bentheimer.raw",
                    "sha256": "…",
                }
            ]
        }, indent=2))
        raise SystemExit(1)

    if not HAVE_REQUESTS:
        raise SystemExit("`requests` required; install with `pip install requests`.")

    manifest = json.loads(manifest_path.read_text())
    for entry in manifest.get("files", []):
        dest = dr / entry["filename"]
        download(entry["url"], dest, entry.get("sha256"))
    print("\nDone.")


if __name__ == "__main__":
    main()
