# Re-running the Kubernetes load test after the overload fix

The Phase 6 load test found pods failing liveness probes under load. The fix is described in
[`results/phase6_overload_fix.md`](../results/phase6_overload_fix.md) and has been measured
locally. This is the procedure for re-measuring it on the k3d cluster. **It has not been executed
yet**, so treat the commands as a checklist to verify as you go, not as a recorded run.

The re-run has to answer three questions:

1. Do liveness probes still fail under the same load? (Expected: no.)
2. Was the original failure caused by the scorer itself or by the node being out of CPU?
3. Does the HPA still scale 2→4?

## 1. Fix the two measurement problems from the first run

**Problem A: `kubectl port-forward` talks to one pod.** When that pod restarted, every later
request failed, which produced the misleading 99.71% failure rate. Send load through the Service.
With k3d, map a host port to the cluster load balancer when creating the cluster:

```bash
k3d cluster delete driftline
k3d cluster create driftline -p "8000:8000@loadbalancer"
docker build -f serving/Dockerfile -t driftline-scorer:latest .
k3d image import driftline-scorer:latest -c driftline
kubectl apply -f k8s/scorer-deployment.yaml
kubectl patch svc driftline-scorer -p '{"spec":{"type":"LoadBalancer"}}'
kubectl rollout status deploy/driftline-scorer
curl -s localhost:8000/readyz
```

(The Deployment mounts model artifacts from a host path. The k3d node needs that path mounted,
as in the original Phase 4 setup.)

**Problem B: the load generator shared CPU with the pods.** Run the load generator from a
second machine if one is available. On a single VM, pin it to cores the cluster does not use,
and record node CPU during the test so starvation is visible:

```bash
cd scripts/loadgen && go build -o /tmp/loadgen . && cd ../..
python scripts/generate_sample_requests.py            # writes serving/sample_requests.json
```

## 2. Run it

Terminal 1, watch the things that went wrong last time:

```bash
kubectl get events -w --field-selector involvedObject.kind=Pod | grep -Ei "unhealthy|killing|backoff"
```

Terminal 2:

```bash
kubectl get hpa driftline-scorer-hpa -w
```

Terminal 3, the load (same 100 clients and 150 s as Phase 6):

```bash
taskset -c 0 /tmp/loadgen -url http://localhost:8000 -health-path /healthz \
    -payloads serving/sample_requests.json -users 100 -duration 150s | tee results/overload/k3d_rerun_100.json
```

## 3. Record

| What | Command | Phase 6 value |
|---|---|---|
| Liveness failures | `kubectl get events \| grep -c "Liveness probe failed"` | nearly every pod |
| Restarts | `kubectl get pods -l app=driftline-scorer` (RESTARTS column) | 1 |
| Why a pod restarted, if any did | `kubectl describe pod <pod>` → "Last State" reason. `OOMKilled` and a liveness kill are different findings | not recorded |
| HPA scale-out | `kubectl describe hpa driftline-scorer-hpa` | 2→4 |
| Requests shed | `curl -s <pod>:8000/metrics \| grep driftline_score_shed_total` | n/a |
| Node CPU during the test | `kubectl top nodes`, or `vmstat 5` on the VM | not recorded |
| Client-side result | the JSON printed by `loadgen` | 99.71% failed (port-forward artifact) |

## 4. Decide

- **No liveness failures, no restarts:** the fix holds. Update the status line in
  `results/phase6_overload_fix.md` and the Phase 6 bullet in the README with the new numbers.
- **Liveness failures remain and node CPU was pinned at 100%:** the node is the bottleneck, not
  the scorer. Re-run with the load generator on another machine before drawing conclusions.
- **A pod shows `OOMKilled`:** a memory problem that this change does not address. Raise the limit
  or profile memory; record it as its own finding.

To test the "starved pod" hypothesis directly, repeat once with `limits.cpu: 250m` on the
Deployment. Locally that is the condition under which the old code failed and the new code did not.
