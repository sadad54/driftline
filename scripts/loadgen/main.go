// Minimal closed-loop load generator for the Driftline scorer, used by
// scripts/probe_starvation_bench.py. Standard library only.
//
// It does two things at once:
//   - N workers POST /score in a loop (backing off briefly when the scorer answers 503);
//   - one prober calls the health endpoint on a fixed period with a 1 s timeout, the way
//     kubelet does with a default probe.
//
// A compiled client matters here: a Python load generator tops out at a few hundred
// requests per second on one core, which is not enough to saturate the scorer, so the
// "overloaded" case never actually happens.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

func pct(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q * float64(len(sorted)))
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return float64(int(sorted[i]*10)) / 10
}

func main() {
	base := flag.String("url", "http://127.0.0.1:8077", "scorer base URL")
	healthPath := flag.String("health-path", "/healthz", "health endpoint to probe")
	payloadFile := flag.String("payloads", "", "JSON array of /score request bodies")
	users := flag.Int("users", 100, "concurrent workers")
	duration := flag.Duration("duration", 30*time.Second, "test length")
	probePeriod := flag.Duration("probe-period", 500*time.Millisecond, "time between probes")
	probeTimeout := flag.Duration("probe-timeout", time.Second, "probe timeout")
	flag.Parse()

	raw, err := os.ReadFile(*payloadFile)
	if err != nil {
		panic(err)
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		panic(err)
	}

	transport := &http.Transport{MaxIdleConns: *users * 2, MaxIdleConnsPerHost: *users * 2}
	client := &http.Client{Transport: transport, Timeout: 30 * time.Second}
	probeClient := &http.Client{Transport: &http.Transport{}, Timeout: *probeTimeout}

	deadline := time.Now().Add(*duration)
	var ok, shed, errs int64
	var mu sync.Mutex
	var latencies []float64
	var wg sync.WaitGroup

	for u := 0; u < *users; u++ {
		wg.Add(1)
		go func(u int) {
			defer wg.Done()
			local := make([]float64, 0, 4096)
			for k := u; time.Now().Before(deadline); k++ {
				t0 := time.Now()
				resp, err := client.Post(*base+"/score", "application/json", bytes.NewReader(items[k%len(items)]))
				if err != nil {
					atomic.AddInt64(&errs, 1)
					continue
				}
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				switch resp.StatusCode {
				case 200:
					atomic.AddInt64(&ok, 1)
					local = append(local, float64(time.Since(t0).Microseconds())/1000)
				case 503:
					atomic.AddInt64(&shed, 1)
					time.Sleep(50 * time.Millisecond)
				default:
					atomic.AddInt64(&errs, 1)
				}
			}
			mu.Lock()
			latencies = append(latencies, local...)
			mu.Unlock()
		}(u)
	}

	var probeLat []float64
	probesFailed := 0
	longestFailRun, failRun := 0, 0
	for time.Now().Before(deadline) {
		t0 := time.Now()
		resp, err := probeClient.Get(*base + *healthPath)
		good := err == nil && resp.StatusCode == 200
		if resp != nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
		if good {
			probeLat = append(probeLat, float64(time.Since(t0).Microseconds())/1000)
			failRun = 0
		} else {
			probesFailed++
			failRun++
			if failRun > longestFailRun {
				longestFailRun = failRun
			}
		}
		if rest := *probePeriod - time.Since(t0); rest > 0 {
			time.Sleep(rest)
		}
	}
	wg.Wait()

	sort.Float64s(latencies)
	sort.Float64s(probeLat)
	probeMax := 0.0
	if len(probeLat) > 0 {
		probeMax = probeLat[len(probeLat)-1]
	}
	out := map[string]any{
		"probes_sent":                 len(probeLat) + probesFailed,
		"probes_failed":               probesFailed,
		"longest_run_of_failed_probes": longestFailRun,
		"probe_p50_ms":                pct(probeLat, 0.50),
		"probe_p99_ms":                pct(probeLat, 0.99),
		"probe_max_ms":                probeMax,
		"score_ok":                    ok,
		"score_shed_503":              shed,
		"score_errors":                errs,
		"score_rps":                   float64(int(float64(ok)/duration.Seconds()*10)) / 10,
		"score_p50_ms":                pct(latencies, 0.50),
		"score_p95_ms":                pct(latencies, 0.95),
		"score_p99_ms":                pct(latencies, 0.99),
	}
	enc, _ := json.MarshalIndent(out, "", "  ")
	fmt.Println(string(enc))
}
