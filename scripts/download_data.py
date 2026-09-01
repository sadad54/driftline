"""Download the three datasets Driftline needs, via the Kaggle API.

Requires ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY env vars) to already
be set up — see driftline/data/raw/README.md for how to get one.

Usage:
    python scripts/download_data.py            # all three
    python scripts/download_data.py --only ieee-cis
"""
import argparse
import subprocess
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = {
    "ieee-cis": {
        "kind": "competition",
        "ref": "ieee-fraud-detection",
        "dest": RAW / "ieee-cis",
    },
    "elliptic": {
        "kind": "dataset",
        "ref": "ellipticco/elliptic-data-set",
        "dest": RAW / "elliptic",
    },
    "paysim": {
        "kind": "dataset",
        "ref": "ealaxi/paysim1",
        "dest": RAW / "paysim",
    },
}


def download(name: str, spec: dict) -> None:
    spec["dest"].mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "competition":
        cmd = ["kaggle", "competitions", "download", "-c", spec["ref"], "-p", str(spec["dest"])]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", spec["ref"], "-p", str(spec["dest"]), "--unzip"]
    print(f"[{name}] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[{name}] FAILED\nstdout: {result.stdout}\nstderr: {result.stderr}", file=sys.stderr)
        if spec["kind"] == "competition" and "403" in result.stderr:
            print(
                f"[{name}] Likely cause: competition rules not accepted yet. "
                f"Visit https://www.kaggle.com/competitions/{spec['ref']}/rules and accept.",
                file=sys.stderr,
            )
        return

    if spec["kind"] == "competition":
        import zipfile

        for zf in spec["dest"].glob("*.zip"):
            print(f"[{name}] unzipping {zf.name}")
            with zipfile.ZipFile(zf) as z:
                z.extractall(spec["dest"])
            zf.unlink()

    print(f"[{name}] done -> {spec['dest']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(DATASETS), default=None)
    args = parser.parse_args()

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS
    for name, spec in targets.items():
        download(name, spec)


if __name__ == "__main__":
    main()
