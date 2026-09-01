"""Create a small, committed sample of IEEE-CIS for the CI quality-gate job.

CI can't download the full 1.3GB dataset (would need Kaggle secrets + several minutes per run
just for data). This writes a small, time-ordered, real sample -- not synthetic data -- small
enough to commit to git and fast enough for a PR-AUC regression check to run in under a minute.
The full research pipeline (all the actual results in this repo) always runs on the complete
590,540-row dataset on the VM; this sample exists ONLY to make CI's quality gate self-contained.
"""
from pathlib import Path

from driftline.data import load_ieee_cis

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
N_ROWS = 6000  # keeps the committed file well under git's soft size limits


def main():
    df = load_ieee_cis()
    sample = df.iloc[:N_ROWS]  # already time-ordered by load_ieee_cis
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ieee_cis_ci_sample.parquet"
    sample.to_parquet(out_path)
    print(f"Wrote {len(sample):,} rows ({sample['isFraud'].mean():.4%} fraud rate) to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
