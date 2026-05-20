#!/usr/bin/env python3
"""
BRUTAL stress-test client for NN Training Monitor Server.

Measures per-request response time (wall-clock), throughput, error rate,
and latency percentiles across multiple test modes:
  single   — sequential, one run
  multi    — N concurrent runs via thread pool
  burst    — fire M requests as fast as possible (no interval), measure saturation
  ramp     — gradually increase concurrency from 1..W, find where latency degrades
  infinite — endless flood (Ctrl+C to stop)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_ENDPOINT = "http://localhost:8000/api/v1/metrics/layerwise"
HEALTH_ENDPOINT = "http://localhost:8000/health"
RUNS_ENDPOINT = "http://localhost:8000/api/v1/runs"
REQUEST_TIMEOUT = 15  # seconds

# ---------------------------------------------------------------------------
# Layer definitions — 221-layer Transformer
# ---------------------------------------------------------------------------

BRUTAL_LAYERS: List[Tuple[str, str]] = []

BRUTAL_LAYERS.extend([
    ("input/embedding", "Embedding"),
    ("input/pos_encoding", "PositionalEncoding"),
    ("input/dropout", "Dropout"),
])

for _bid in range(1, 13):
    BRUTAL_LAYERS.extend([
        (f"encoder/block{_bid}/norm1", "LayerNorm"),
        (f"encoder/block{_bid}/attn/in_proj", "Linear"),
        (f"encoder/block{_bid}/attn/qkv_split", "Split"),
        (f"encoder/block{_bid}/attn/q_proj", "Linear"),
        (f"encoder/block{_bid}/attn/k_proj", "Linear"),
        (f"encoder/block{_bid}/attn/v_proj", "Linear"),
        (f"encoder/block{_bid}/attn/softmax", "Softmax"),
        (f"encoder/block{_bid}/attn/out_proj", "Linear"),
        (f"encoder/block{_bid}/attn/dropout", "Dropout"),
        (f"encoder/block{_bid}/resid_add1", "Add"),
        (f"encoder/block{_bid}/norm2", "LayerNorm"),
        (f"encoder/block{_bid}/ffn/linear1", "Linear"),
        (f"encoder/block{_bid}/ffn/activation", "GELU"),
        (f"encoder/block{_bid}/ffn/dropout1", "Dropout"),
        (f"encoder/block{_bid}/ffn/linear2", "Linear"),
        (f"encoder/block{_bid}/ffn/dropout2", "Dropout"),
        (f"encoder/block{_bid}/resid_add2", "Add"),
    ])

BRUTAL_LAYERS.extend([
    ("final/norm", "LayerNorm"),
    ("pooler/mean", "MeanPooling"),
    ("pooler/tanh", "Tanh"),
    ("classifier/linear1", "Linear"),
    ("classifier/norm1", "BatchNorm1d"),
    ("classifier/activation1", "ReLU"),
    ("classifier/dropout1", "Dropout"),
    ("classifier/linear2", "Linear"),
    ("classifier/norm2", "BatchNorm1d"),
    ("classifier/activation2", "ReLU"),
    ("classifier/dropout2", "Dropout"),
    ("classifier/linear3", "Linear"),
    ("classifier/output", "Linear"),
    ("classifier/softmax", "Softmax"),
])

# ---------------------------------------------------------------------------
# Per-request metric record
# ---------------------------------------------------------------------------


@dataclass
class RequestMetric:
    step: int
    run_id: str
    ok: bool
    status_code: Optional[int]
    elapsed_ms: float          # wall-clock round-trip (ms)
    payload_bytes: int
    alerts: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Aggregated results
# ---------------------------------------------------------------------------


@dataclass
class StressTestResult:
    run_ids: List[str] = field(default_factory=list)
    metrics: List[RequestMetric] = field(default_factory=list)
    wall_start: float = 0.0
    wall_end: float = 0.0

    # -- helpers -------------------------------------------------------------

    @property
    def wall_seconds(self) -> float:
        return self.wall_end - self.wall_start

    @property
    def total_requests(self) -> int:
        return len(self.metrics)

    @property
    def ok_requests(self) -> int:
        return sum(1 for m in self.metrics if m.ok)

    @property
    def fail_requests(self) -> int:
        return self.total_requests - self.ok_requests

    @property
    def error_rate_pct(self) -> float:
        return (self.fail_requests / self.total_requests * 100) if self.total_requests else 0.0

    @property
    def throughput_rps(self) -> float:
        return self.ok_requests / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def total_alerts(self) -> int:
        return sum(m.alerts for m in self.metrics)

    @property
    def total_payload_mb(self) -> float:
        return sum(m.payload_bytes for m in self.metrics) / (1024 * 1024)

    def _latencies(self) -> List[float]:
        return [m.elapsed_ms for m in self.metrics if m.ok]

    def percentile(self, p: float) -> Optional[float]:
        lats = self._latencies()
        if not lats:
            return None
        s = sorted(lats)
        idx = (p / 100) * (len(s) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] + frac * (s[hi] - s[lo])

    @property
    def min_ms(self) -> Optional[float]:
        lats = self._latencies()
        return min(lats) if lats else None

    @property
    def max_ms(self) -> Optional[float]:
        lats = self._latencies()
        return max(lats) if lats else None

    @property
    def mean_ms(self) -> Optional[float]:
        lats = self._latencies()
        return statistics.mean(lats) if lats else None

    @property
    def median_ms(self) -> Optional[float]:
        lats = self._latencies()
        return statistics.median(lats) if lats else None

    @property
    def stdev_ms(self) -> Optional[float]:
        lats = self._latencies()
        return statistics.stdev(lats) if len(lats) >= 2 else None

    def status_histogram(self) -> Dict[str, int]:
        h: Dict[str, int] = defaultdict(int)
        for m in self.metrics:
            key = str(m.status_code) if m.status_code else "timeout/err"
            h[key] += 1
        return dict(sorted(h.items()))

    def error_messages(self, limit: int = 5) -> List[str]:
        errs = [m.error for m in self.metrics if m.error]
        return errs[:limit]

    # -- printing ------------------------------------------------------------

    def print_summary(self, title: str = "STRESS TEST RESULTS") -> None:
        print()
        print("=" * 72)
        print(f" {title}")
        print("=" * 72)
        print(f"  Duration          : {self.wall_seconds:.2f} s")
        print(f"  Run IDs           : {', '.join(self.run_ids)}")
        print(f"  Layers / payload  : {len(BRUTAL_LAYERS)} layers  "
              f"~{self.metrics[0].payload_bytes / 1024:.1f} KB/req" if self.metrics else "")
        print()
        print(f"  Requests total    : {self.total_requests}")
        print(f"  Requests OK       : {self.ok_requests}")
        print(f"  Requests failed   : {self.fail_requests}")
        print(f"  Error rate        : {self.error_rate_pct:.1f}%")
        print(f"  Status codes      : {self.status_histogram()}")
        print()
        print(f"  Throughput        : {self.throughput_rps:.1f} req/s")
        print(f"  Total data sent   : {self.total_payload_mb:.2f} MB")
        print(f"  Total alerts      : {self.total_alerts}")
        print()

        lats = self._latencies()
        if lats:
            print("  Response time (ms):")
            print(f"    min     : {self.min_ms:.2f}")
            print(f"    mean    : {self.mean_ms:.2f}")
            print(f"    median  : {self.median_ms:.2f}")
            if self.stdev_ms is not None:
                print(f"    stdev   : {self.stdev_ms:.2f}")
            print(f"    max     : {self.max_ms:.2f}")
            print(f"    p90     : {self.percentile(90):.2f}")
            print(f"    p95     : {self.percentile(95):.2f}")
            print(f"    p99     : {self.percentile(99):.2f}")

        errs = self.error_messages()
        if errs:
            print()
            print(f"  First errors ({len(errs)} shown):")
            for e in errs:
                print(f"    - {e}")

        print("=" * 72)
        print()

    def print_csv(self) -> None:
        print("step,run_id,ok,status_code,elapsed_ms,payload_bytes,alerts,error")
        for m in self.metrics:
            err_str = (m.error or "").replace('"', '""')
            print(
                f'{m.step},{m.run_id},{m.ok},{m.status_code},'
                f'{m.elapsed_ms:.3f},{m.payload_bytes},{m.alerts},'
                f'"{err_str}"'
            )


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def generate_brutal_layer_stats(
    layer_id: str,
    layer_type: str,
    depth_index: int,
    prev_std: Optional[float] = None,
    severity: str = "extreme",
) -> Dict[str, Any]:
    activation_std = random.uniform(0.01, 0.8)
    gradient_l2_norm = random.uniform(0.0001, 0.1)
    gradient_max_abs = gradient_l2_norm * random.uniform(1.0, 3.0)

    if severity == "extreme":
        if random.random() < 0.8:
            gradient_l2_norm = random.uniform(0.00001, 0.0009)
            gradient_max_abs = gradient_l2_norm * random.uniform(1.0, 2.0)
        if random.random() < 0.4:
            gradient_l2_norm = random.uniform(50.0, 500.0)
            gradient_max_abs = random.uniform(100.0, 1000.0)
        if random.random() < 0.6:
            activation_std = random.uniform(0.0001, 0.001)
    elif severity == "vanishing":
        gradient_l2_norm = random.uniform(0.00001, 0.0005)
        gradient_max_abs = random.uniform(0.0001, 0.001)
        activation_std = random.uniform(0.0001, 0.01)
    elif severity == "exploding":
        gradient_l2_norm = random.uniform(100.0, 1000.0)
        gradient_max_abs = random.uniform(500.0, 5000.0)
        activation_std = random.uniform(5.0, 50.0)

    cross_layer_ratio = None
    if prev_std is not None:
        if random.random() < 0.7:
            cross_layer_ratio = random.uniform(0.001, 0.09)
        else:
            cross_layer_ratio = activation_std / (prev_std + 1e-8)

    return {
        "layer_id": layer_id,
        "layer_type": layer_type,
        "depth_index": depth_index,
        "intermediate_features": {
            "activation_std": activation_std,
            "activation_mean": random.uniform(-0.5, 0.5),
            "activation_shape": [random.randint(64, 256), random.randint(128, 1024)],
            "cross_layer_std_ratio": cross_layer_ratio,
        },
        "gradient_flow": {
            "gradient_l2_norm": gradient_l2_norm,
            "gradient_std": gradient_l2_norm * random.uniform(0.001, 0.1),
            "gradient_max_abs": gradient_max_abs,
        },
        "parameter_statistics": {
            "weight": {
                "std": random.uniform(0.001, 0.5),
                "mean": random.uniform(-0.1, 0.1),
                "spectral_norm": random.uniform(0.5, 10.0),
                "frobenius_norm": random.uniform(1.0, 50.0),
            },
            "bias": {
                "std": random.uniform(0.001, 0.1),
                "mean_abs": random.uniform(0.001, 0.1),
            },
        },
    }


def count_brutal_alerts(layers: List[Dict]) -> int:
    count = 0
    for layer in layers:
        grad_norm = layer["gradient_flow"]["gradient_l2_norm"]
        grad_max = layer["gradient_flow"]["gradient_max_abs"]
        ratio = layer["intermediate_features"].get("cross_layer_std_ratio")
        if grad_norm < 0.001:
            count += 1
        if grad_max > 10:
            count += 1
        if ratio is not None and ratio < 0.1:
            count += 1
    return count


def _build_payload(run_id: str, global_step: int, severity: str) -> Tuple[Dict[str, Any], int]:
    """Build the JSON payload and return (payload_dict, alert_count)."""
    layer_statistics = []
    prev_std = None
    prev_grad_norm = None
    gradient_norm_ratios: Dict[str, float] = {}

    for depth_idx, (layer_id, layer_type) in enumerate(BRUTAL_LAYERS):
        layer_data = generate_brutal_layer_stats(
            layer_id, layer_type, depth_idx, prev_std, severity,
        )
        layer_statistics.append(layer_data)
        current_std = layer_data["intermediate_features"]["activation_std"]
        current_grad_norm = layer_data["gradient_flow"]["gradient_l2_norm"]
        if prev_grad_norm is not None and prev_grad_norm > 0:
            ratio = current_grad_norm / prev_grad_norm
            gradient_norm_ratios[f"{layer_id}_to_prev"] = ratio
        prev_std = current_std
        prev_grad_norm = current_grad_norm

    stds = [l["intermediate_features"]["activation_std"] for l in layer_statistics]
    feature_std_gradient = (stds[-1] - stds[0]) / len(stds) if len(stds) > 1 else -0.5
    alerts = count_brutal_alerts(layer_statistics)

    layer_groups: Dict[str, List[str]] = {
        "input": ["input/embedding", "input/pos_encoding", "input/dropout"],
    }
    for bid in range(1, 13):
        prefix = f"encoder/block{bid}"
        layer_groups[prefix] = [
            f"{prefix}/norm1", f"{prefix}/attn/in_proj",
            f"{prefix}/attn/qkv_split", f"{prefix}/attn/q_proj",
            f"{prefix}/attn/k_proj", f"{prefix}/attn/v_proj",
            f"{prefix}/attn/softmax", f"{prefix}/attn/out_proj",
            f"{prefix}/attn/dropout", f"{prefix}/resid_add1",
            f"{prefix}/norm2", f"{prefix}/ffn/linear1",
            f"{prefix}/ffn/activation", f"{prefix}/ffn/dropout1",
            f"{prefix}/ffn/linear2", f"{prefix}/ffn/dropout2",
            f"{prefix}/resid_add2",
        ]
    layer_groups["final"] = ["final/norm", "pooler/mean", "pooler/tanh"]
    layer_groups["classifier"] = [
        "classifier/linear1", "classifier/norm1", "classifier/activation1",
        "classifier/dropout1", "classifier/linear2", "classifier/norm2",
        "classifier/activation2", "classifier/dropout2", "classifier/linear3",
        "classifier/output", "classifier/softmax",
    ]

    payload = {
        "metadata": {
            "run_id": run_id,
            "timestamp": time.time(),
            "global_step": global_step,
            "batch_size": 128,
            "layer_groups": layer_groups,
        },
        "layer_statistics": layer_statistics,
        "cross_layer_analysis": {
            "feature_std_gradient": feature_std_gradient,
            "gradient_norm_ratio": gradient_norm_ratios,
        },
    }
    return payload, alerts


# ---------------------------------------------------------------------------
# Single request sender (timed)
# ---------------------------------------------------------------------------

_session = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_session, "s"):
        _session.s = requests.Session()
    return _session.s


def send_one(
    run_id: str,
    step: int,
    severity: str = "extreme",
    endpoint: str = API_ENDPOINT,
) -> RequestMetric:
    payload, alerts = _build_payload(run_id, step, severity)
    payload_bytes = len(json.dumps(payload).encode())

    t0 = time.perf_counter()
    try:
        resp = _get_session().post(endpoint, json=payload, timeout=REQUEST_TIMEOUT)
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code == 202:
            return RequestMetric(
                step=step, run_id=run_id, ok=True,
                status_code=resp.status_code, elapsed_ms=elapsed,
                payload_bytes=payload_bytes, alerts=alerts,
            )
        return RequestMetric(
            step=step, run_id=run_id, ok=False,
            status_code=resp.status_code, elapsed_ms=elapsed,
            payload_bytes=payload_bytes, alerts=0,
            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return RequestMetric(
            step=step, run_id=run_id, ok=False,
            status_code=None, elapsed_ms=elapsed,
            payload_bytes=payload_bytes, alerts=0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_health(endpoint_base: str) -> bool:
    url = endpoint_base.rstrip("/") + "/health"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  Server healthy: {json.dumps(data)}")
            return True
        print(f"  Server responded {r.status_code}")
        return False
    except Exception as e:
        print(f"  Cannot reach server at {url}: {e}")
        return False


# ---------------------------------------------------------------------------
# Test modes
# ---------------------------------------------------------------------------


def run_single(
    run_id: str,
    steps: int,
    interval: float,
    endpoint: str,
) -> StressTestResult:
    result = StressTestResult(run_ids=[run_id])
    severities = ["extreme", "vanishing", "exploding"]

    print(f"\n{'='*72}")
    print(f" SINGLE-RUN STRESS: {run_id}")
    print(f" Layers: {len(BRUTAL_LAYERS)}  |  Steps: {steps}  |  Interval: {interval}s")
    print(f"{'='*72}\n")

    result.wall_start = time.perf_counter()

    for step in range(steps):
        severity = severities[step % len(severities)]
        m = send_one(run_id, step, severity, endpoint)
        result.metrics.append(m)

        if m.ok:
            sym = "+" if m.elapsed_ms < 50 else "!" if m.elapsed_ms < 200 else "SLOW"
            print(
                f"  [{sym}] Step {step:4d}  {m.elapsed_ms:7.1f} ms  "
                f"alerts={m.alerts:4d}  status={m.status_code}"
            )
        else:
            print(
                f"  [X] Step {step:4d}  {m.elapsed_ms:7.1f} ms  "
                f"FAIL  status={m.status_code}  err={m.error}"
            )

        if step < steps - 1:
            time.sleep(interval)

    result.wall_end = time.perf_counter()
    return result


def run_multi(
    num_runs: int,
    steps_per_run: int,
    endpoint: str,
) -> StressTestResult:
    result = StressTestResult()

    print(f"\n{'#'*72}")
    print(f" MULTI-RUN ASSAULT: {num_runs} runs x {steps_per_run} steps")
    print(f"{'#'*72}\n")

    barrier = threading.Barrier(num_runs, timeout=30)
    per_run_results: List[StressTestResult] = []

    def _worker(run_idx: int) -> StressTestResult:
        rid = f"BRUTAL_{run_idx}_{int(time.time())}"
        res = StressTestResult(run_ids=[rid])
        barrier.wait()  # all threads start together
        for step in range(steps_per_run):
            sev = ["extreme", "vanishing", "exploding"][step % 3]
            m = send_one(rid, step, sev, endpoint)
            res.metrics.append(m)
        res.wall_start = result.wall_start
        res.wall_end = time.perf_counter()
        return res

    result.wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_runs) as pool:
        futures = [pool.submit(_worker, i) for i in range(num_runs)]
        for f in as_completed(futures):
            per_run_results.append(f.result())

    result.wall_end = time.perf_counter()

    for rr in per_run_results:
        result.run_ids.extend(rr.run_ids)
        result.metrics.extend(rr.metrics)

    return result


def run_burst(
    run_id: str,
    total_requests: int,
    endpoint: str,
) -> StressTestResult:
    """Fire requests as fast as possible with no interval."""
    result = StressTestResult(run_ids=[run_id])

    print(f"\n{'*'*72}")
    print(f" BURST MODE: {total_requests} requests, zero delay")
    print(f"{'*'*72}\n")

    result.wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=64) as pool:
        futs = {
            pool.submit(send_one, run_id, i, "extreme", endpoint): i
            for i in range(total_requests)
        }
        done_count = 0
        for f in as_completed(futs):
            m = f.result()
            result.metrics.append(m)
            done_count += 1
            if done_count % max(1, total_requests // 20) == 0 or done_count == total_requests:
                ok_so_far = sum(1 for x in result.metrics if x.ok)
                print(
                    f"  Progress: {done_count}/{total_requests}  "
                    f"ok={ok_so_far}  fail={done_count - ok_so_far}"
                )

    result.wall_end = time.perf_counter()
    # sort metrics by step for clean percentile calc
    result.metrics.sort(key=lambda m: m.step)
    return result


def run_ramp(
    run_id: str,
    steps_per_worker: int,
    max_workers: int,
    endpoint: str,
) -> StressTestResult:
    """Gradually increase concurrency 1..max_workers, measure latency at each level."""
    result = StressTestResult(run_ids=[run_id])

    print(f"\n{'~'*72}")
    print(f" RAMP-UP TEST: 1..{max_workers} workers, {steps_per_worker} req/worker")
    print(f"{'~'*72}\n")

    level_results: List[StressTestResult] = []
    overall_start = time.perf_counter()

    for w in range(1, max_workers + 1):
        level = StressTestResult()
        level.wall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=w) as pool:
            futs = [pool.submit(send_one, run_id, i, "extreme", endpoint)
                    for i in range(steps_per_worker)]
            for f in as_completed(futs):
                level.metrics.append(f.result())

        level.wall_end = time.perf_counter()
        level.run_ids = [run_id]
        level_results.append(level)

        ok = level.ok_requests
        mean_lat = level.mean_ms or 0
        p95_lat = level.percentile(95) or 0
        rps = level.throughput_rps
        print(
            f"  Workers={w:2d}  "
            f"ok={ok}/{level.total_requests}  "
            f"mean={mean_lat:7.1f}ms  "
            f"p95={p95_lat:7.1f}ms  "
            f"rps={rps:6.1f}"
        )

    result.wall_start = overall_start
    result.wall_end = time.perf_counter()
    for lr in level_results:
        result.metrics.extend(lr.metrics)

    # print breakdown table
    print(f"\n  {'Workers':>7}  {'OK':>5}  {'Fail':>5}  "
          f"{'Mean ms':>9}  {'P50 ms':>9}  {'P95 ms':>9}  {'P99 ms':>9}  {'RPS':>8}")
    print(f"  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")
    for i, lr in enumerate(level_results):
        w = i + 1
        print(
            f"  {w:>7}  {lr.ok_requests:>5}  {lr.fail_requests:>5}  "
            f"{(lr.mean_ms or 0):>9.1f}  {(lr.percentile(50) or 0):>9.1f}  "
            f"{(lr.percentile(95) or 0):>9.1f}  {(lr.percentile(99) or 0):>9.1f}  "
            f"{lr.throughput_rps:>8.1f}"
        )

    return result


def run_infinite(
    run_id: str,
    endpoint: str,
) -> None:
    print(f"\n{'>'*72}")
    print(f" INFINITE FLOOD: {run_id}  (Ctrl+C to stop)")
    print(f"{'>'*72}\n")

    step = 0
    latencies: List[float] = []
    total_alerts = 0

    try:
        while True:
            m = send_one(run_id, step, "extreme", endpoint)
            latencies.append(m.elapsed_ms)

            if m.ok:
                total_alerts += m.alerts
                # rolling stats every 100 requests
                if step > 0 and step % 100 == 0:
                    recent = latencies[-100:]
                    print(
                        f"  Step {step:6d}  "
                        f"alerts={total_alerts:8d}  "
                        f"last={m.elapsed_ms:.1f}ms  "
                        f"mean100={statistics.mean(recent):.1f}ms  "
                        f"max100={max(recent):.1f}ms"
                    )
                else:
                    sym = "+" if m.elapsed_ms < 50 else "!" if m.elapsed_ms < 200 else "SLOW"
                    print(f"  [{sym}] Step {step:6d}  {m.elapsed_ms:7.1f} ms  alerts={m.alerts}")
            else:
                print(f"  [X] Step {step:6d}  FAIL  {m.error}")

            step += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped after {step} requests, {total_alerts} alerts")
        if latencies:
            print(f"  Latency  mean={statistics.mean(latencies):.1f}ms  "
                  f"median={statistics.median(latencies):.1f}ms  "
                  f"max={max(latencies):.1f}ms")


# ---------------------------------------------------------------------------
# Verify server stored the data
# ---------------------------------------------------------------------------


def verify_runs(run_ids: List[str], endpoint_base: str) -> None:
    url = endpoint_base.rstrip("/") + "/api/v1/runs"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            print(f"  Could not verify runs: HTTP {r.status_code}")
            return
        data = r.json()
        server_runs = set()
        if isinstance(data, dict):
            server_runs = set(data.keys())
        elif isinstance(data, list):
            server_runs = {item.get("run_id", item.get("id")) for item in data}

        found = server_runs & set(run_ids)
        print(f"  Server /api/v1/runs returned {len(server_runs)} runs; "
              f"{len(found)}/{len(run_ids)} test runs found")
    except Exception as e:
        print(f"  Verification request failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BRUTAL stress test with response-time measurements for NN Monitor Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  single   — sequential, one run, fixed interval\n"
            "  multi    — N concurrent runs via thread pool\n"
            "  burst    — fire M requests as fast as possible (no interval)\n"
            "  ramp     — increase concurrency 1..W, measure latency per level\n"
            "  infinite — endless flood (Ctrl+C to stop)\n"
        ),
    )
    parser.add_argument(
        "--mode", choices=["single", "multi", "burst", "ramp", "infinite"],
        default="single",
        help="Test mode (default: single)",
    )
    parser.add_argument("--run-id", default=f"BRUTAL_{int(time.time())}", help="Run identifier")
    parser.add_argument("--steps", type=int, default=100, help="Steps (single/burst mode)")
    parser.add_argument("--runs", type=int, default=3, help="Parallel runs (multi mode)")
    parser.add_argument(
        "--interval", type=float, default=0.1, help="Seconds between requests (single mode)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=16, help="Max concurrency (ramp mode)",
    )
    parser.add_argument(
        "--steps-per-worker", type=int, default=50, help="Requests per worker (ramp mode)",
    )
    parser.add_argument(
        "--endpoint", default=API_ENDPOINT, help="Metrics API endpoint URL",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Server base URL (for health/runs)",
    )
    parser.add_argument(
        "--csv", action="store_true", help="Also output per-request CSV to stdout",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip post-test /runs verification",
    )

    args = parser.parse_args()

    base_url = args.base_url
    endpoint = args.endpoint

    print()
    print("-" * 72)
    print(f" Server: {base_url}")
    print(f" Endpoint: {endpoint}")
    print(f" Layers: {len(BRUTAL_LAYERS)}")
    print(f" Mode: {args.mode}")
    print("-" * 72)

    # Health check
    print("\n  Pre-flight health check...")
    if not check_health(base_url):
        print("  ABORT: server not reachable.\n")
        sys.exit(1)

    # Run selected mode
    result: Optional[StressTestResult] = None

    if args.mode == "single":
        result = run_single(args.run_id, args.steps, args.interval, endpoint)

    elif args.mode == "multi":
        result = run_multi(args.runs, args.steps, endpoint)

    elif args.mode == "burst":
        result = run_burst(args.run_id, args.steps, endpoint)

    elif args.mode == "ramp":
        result = run_ramp(args.run_id, args.steps_per_worker, args.max_workers, endpoint)

    elif args.mode == "infinite":
        run_infinite(args.run_id, endpoint)
        return  # no summary for infinite

    # Summary
    if result:
        result.print_summary(title=f"STRESS TEST RESULTS — {args.mode.upper()}")

        if not args.no_verify and result.run_ids:
            print("  Post-test verification...")
            verify_runs(result.run_ids, base_url)
            print()

        if args.csv:
            result.print_csv()


if __name__ == "__main__":
    main()
