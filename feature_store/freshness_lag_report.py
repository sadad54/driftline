"""Report p50/p95/p99 feature freshness lag (event produced_at -> online-store write time),
per window size, from the samples streaming/velocity_aggregator.py's RedisVelocityMap collected
into Redis lists during the run.
"""
import numpy as np
import redis

WINDOWS = ["1h", "24h", "7d"]


def main():
    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    for label in WINDOWS:
        raw = client.lrange(f"freshness_lags:{label}", 0, -1)
        if not raw:
            print(f"{label}: no samples")
            continue
        lags = np.array([float(x) for x in raw])
        print(f"{label}: n={len(lags)}  p50={np.percentile(lags, 50):.3f}s  "
              f"p95={np.percentile(lags, 95):.3f}s  p99={np.percentile(lags, 99):.3f}s  "
              f"max={lags.max():.3f}s")


if __name__ == "__main__":
    main()
