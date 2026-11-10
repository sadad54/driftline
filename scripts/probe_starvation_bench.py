"""Local overload benchmark for the scorer: does a saturated pod still answer its health probe?

Background: the Phase 6 load test saw scorer pods fail liveness probes under load and one get
restarted by kubelet (results/phase6_production.md). This script recreates the conditions of one
pod on one machine, so the question can be answered without a cluster.

What it does:
  1. Starts the scorer as a single uvicorn process under a CPU limit. The limit is applied the way
     Kubernetes applies `limits.cpu`: a CFS quota on a cgroup, so the process still sees every
     host core but is throttled once it has used its share of each 100 ms period. This needs a
     writable cgroup (root on Linux). Without one the script pins the server to one core, which
     limits CPU but does not reproduce throttling.
  2. Drives /score with N concurrent clients using real rows from the committed CI sample, and at
     the same time calls the health endpoint every 0.5 s with a 1 s timeout -- what kubelet does
     with a default probe. Both are done by the small Go client in scripts/loadgen (a Python
     client cannot generate enough load to saturate the scorer).
  3. Reports failed probes, probe latency, scoring latency, shed (503) responses, and how many
     CFS periods the kernel throttled the server in.

What it is not: a Kubernetes test. There is no kubelet, no HPA and no neighbouring pods. The
cluster re-run is described in docs/overload-rerun.md.

Usage (repo root, after `python scripts/export_onnx.py --fixture`; needs Go 1.21+):
    python scripts/probe_starvation_bench.py --users 100 --duration 30
    python scripts/probe_starvation_bench.py --repo ../driftline-prefix --health-path /health
        (benchmarks another checkout, e.g. a git worktree of the commit before the fix)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from driftline.data import feature_columns

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "ieee_cis_ci_sample.parquet"
LOADGEN_DIR = REPO / "scripts" / "loadgen"
CGROUP_V1 = Path("/sys/fs/cgroup/cpu")
CGROUP_V2 = Path("/sys/fs/cgroup")
CFS_PERIOD_US = 100_000


def build_payloads(n: int = 200) -> list[dict]:
    """Real transaction rows from the committed CI sample, shaped like /score requests."""
    df = pd.read_parquet(FIXTURE)
    cols = feature_columns(df)
    payloads = []
    for _, row in df.sample(n=min(n, len(df)), random_state=0).iterrows():
        features = {}
        for col in cols:
            val = row[col]
            if pd.isna(val):
                features[col] = None
            elif isinstance(val, str):
                features[col] = val
            else:
                features[col] = float(val)
        payloads.append({"features": features})
    return payloads


def limit_with_cgroup(pid: int, cpus: float) -> str | None:
    """Put `pid` in a fresh cgroup with a CFS quota of `cpus` cores. Returns the mode used, or
    None if no cgroup could be written (not root, not Linux, or a read-only hierarchy)."""
    quota = int(cpus * CFS_PERIOD_US)
    name = f"driftline-bench-{pid}"
    try:
        if (CGROUP_V2 / "cgroup.controllers").exists():
            group = CGROUP_V2 / name
            group.mkdir()
            (group / "cpu.max").write_text(f"{quota} {CFS_PERIOD_US}")
            (group / "cgroup.procs").write_text(str(pid))
            return "cgroup-v2-quota"
        if CGROUP_V1.exists():
            group = CGROUP_V1 / name
            group.mkdir()
            (group / "cpu.cfs_period_us").write_text(str(CFS_PERIOD_US))
            (group / "cpu.cfs_quota_us").write_text(str(quota))
            (group / "cgroup.procs").write_text(str(pid))
            return "cgroup-v1-quota"
    except OSError:
        return None
    return None


def throttle_stats(pid: int) -> dict:
    """How often the kernel throttled the server (cgroup v1 or v2 cpu.stat)."""
    for base in (CGROUP_V1, CGROUP_V2):
        stat = base / f"driftline-bench-{pid}" / "cpu.stat"
        if stat.exists():
            fields = dict(line.split() for line in stat.read_text().splitlines())
            return {"periods": int(fields.get("nr_periods", 0)), "throttled": int(fields.get("nr_throttled", 0))}
    return {}


def remove_cgroup(pid: int) -> None:
    for base in (CGROUP_V1, CGROUP_V2):
        group = base / f"driftline-bench-{pid}"
        if group.exists():
            try:
                group.rmdir()
            except OSError:
                pass


def wait_until_up(url: str) -> None:
    for _ in range(150):
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    raise RuntimeError("scorer did not start")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO), help="checkout whose serving/app.py to benchmark")
    parser.add_argument("--health-path", default="/healthz")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--cpu-limit", type=float, default=1.0, help="server CPU limit in cores")
    parser.add_argument("--uvicorn-arg", action="append", default=[], help="extra uvicorn argument (repeatable)")
    parser.add_argument("--loadgen-bin", default=None, help="prebuilt scripts/loadgen binary (default: go run)")
    parser.add_argument("--out", default=None, help="write the result JSON here")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(build_payloads(), f)
        payload_path = f.name

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", str(args.port),
         "--log-level", "warning", *args.uvicorn_arg],
        cwd=args.repo,
    )
    limit_mode = limit_with_cgroup(server.pid, args.cpu_limit)
    if limit_mode is None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(server.pid, {min(os.sched_getaffinity(0))})
        limit_mode = "pinned-to-one-core"
    try:
        wait_until_up(base_url + args.health_path)
        loadgen = [args.loadgen_bin] if args.loadgen_bin else ["go", "run", "."]
        proc = subprocess.run(
            [*loadgen, "-url", base_url, "-health-path", args.health_path, "-payloads", payload_path,
             "-users", str(args.users), "-duration", f"{args.duration}s"],
            cwd=LOADGEN_DIR, capture_output=True, text=True, check=True,
        )
        measured = json.loads(proc.stdout)
        alive_after = server.poll() is None
        throttling = throttle_stats(server.pid)
    finally:
        server.terminate()
        server.wait(timeout=10)
        remove_cgroup(server.pid)
        os.unlink(payload_path)

    summary = {
        "checkout": Path(args.repo).resolve().name,
        "health_path": args.health_path,
        "users": args.users,
        "duration_s": args.duration,
        "cpu_limit_cores": args.cpu_limit,
        "cpu_limit_mode": limit_mode,
        "uvicorn_args": args.uvicorn_arg,
        "cfs_periods": throttling.get("periods"),
        "cfs_periods_throttled": throttling.get("throttled"),
        "probe_timeout_s": 1.0,
        **measured,
        "server_alive_after": alive_after,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
