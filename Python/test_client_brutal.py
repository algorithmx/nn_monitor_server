#!/usr/bin/env python3
"""
BRUTAL test client for NN Training Monitor Server
Generates MASSIVE amounts of alerts to stress test the frontend UI
"""

import requests
import time
import random
import threading
import argparse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

API_ENDPOINT = "http://localhost:8000/api/v1/metrics/layerwise"

# MASSIVE deep network - 100+ layers to generate huge alert volume
BRUTAL_LAYERS = []

# Input block
BRUTAL_LAYERS.extend([
    ("input/embedding", "Embedding"),
    ("input/pos_encoding", "PositionalEncoding"),
    ("input/dropout", "Dropout"),
])

# 12 Transformer blocks with 10 layers each = 120 layers
for block_id in range(1, 13):
    BRUTAL_LAYERS.extend([
        (f"encoder/block{block_id}/norm1", "LayerNorm"),
        (f"encoder/block{block_id}/attn/in_proj", "Linear"),
        (f"encoder/block{block_id}/attn/qkv_split", "Split"),
        (f"encoder/block{block_id}/attn/q_proj", "Linear"),
        (f"encoder/block{block_id}/attn/k_proj", "Linear"),
        (f"encoder/block{block_id}/attn/v_proj", "Linear"),
        (f"encoder/block{block_id}/attn/softmax", "Softmax"),
        (f"encoder/block{block_id}/attn/out_proj", "Linear"),
        (f"encoder/block{block_id}/attn/dropout", "Dropout"),
        (f"encoder/block{block_id}/resid_add1", "Add"),
        (f"encoder/block{block_id}/norm2", "LayerNorm"),
        (f"encoder/block{block_id}/ffn/linear1", "Linear"),
        (f"encoder/block{block_id}/ffn/activation", "GELU"),
        (f"encoder/block{block_id}/ffn/dropout1", "Dropout"),
        (f"encoder/block{block_id}/ffn/linear2", "Linear"),
        (f"encoder/block{block_id}/ffn/dropout2", "Dropout"),
        (f"encoder/block{block_id}/resid_add2", "Add"),
    ])

# Pooler and classifier
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


def generate_brutal_layer_stats(
    layer_id: str,
    layer_type: str,
    depth_index: int,
    prev_std: Optional[float] = None,
    severity: str = "extreme"
) -> Dict[str, Any]:
    """Generate layer statistics with EXTREME problems"""
    
    # Default to problematic values
    activation_std = random.uniform(0.01, 0.8)
    gradient_l2_norm = random.uniform(0.0001, 0.1)
    gradient_max_abs = gradient_l2_norm * random.uniform(1.0, 3.0)
    
    # Force EXTREME problems based on severity
    if severity == "extreme":
        # 80% chance of vanishing gradients
        if random.random() < 0.8:
            gradient_l2_norm = random.uniform(0.00001, 0.0009)
            gradient_max_abs = gradient_l2_norm * random.uniform(1.0, 2.0)
        
        # 40% chance of exploding
        if random.random() < 0.4:
            gradient_l2_norm = random.uniform(50.0, 500.0)
            gradient_max_abs = random.uniform(100.0, 1000.0)
        
        # 60% chance of dead neurons
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
    
    # Force severe activation drops
    cross_layer_ratio = None
    if prev_std is not None:
        # 70% chance of severe drop
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
            "cross_layer_std_ratio": cross_layer_ratio
        },
        "gradient_flow": {
            "gradient_l2_norm": gradient_l2_norm,
            "gradient_std": gradient_l2_norm * random.uniform(0.001, 0.1),
            "gradient_max_abs": gradient_max_abs
        },
        "parameter_statistics": {
            "weight": {
                "std": random.uniform(0.001, 0.5),
                "mean": random.uniform(-0.1, 0.1),
                "spectral_norm": random.uniform(0.5, 10.0),
                "frobenius_norm": random.uniform(1.0, 50.0)
            },
            "bias": {
                "std": random.uniform(0.001, 0.1),
                "mean_abs": random.uniform(0.001, 0.1)
            }
        }
    }


def send_brutal_metrics(
    run_id: str,
    global_step: int,
    batch_size: int = 128,
    severity: str = "extreme"
) -> tuple[bool, int]:
    """Send metrics with EXTREME problems"""
    
    layer_statistics = []
    prev_std = None
    prev_grad_norm = None
    gradient_norm_ratios = {}
    
    for depth_idx, (layer_id, layer_type) in enumerate(BRUTAL_LAYERS):
        layer_data = generate_brutal_layer_stats(
            layer_id, layer_type, depth_idx, prev_std, severity
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
    
    # Build layer groups
    layer_groups = {
        "input": ["input/embedding", "input/pos_encoding", "input/dropout"],
    }
    
    for block_id in range(1, 13):
        layer_groups[f"encoder/block{block_id}"] = [
            f"encoder/block{block_id}/norm1",
            f"encoder/block{block_id}/attn/in_proj",
            f"encoder/block{block_id}/attn/qkv_split",
            f"encoder/block{block_id}/attn/q_proj",
            f"encoder/block{block_id}/attn/k_proj",
            f"encoder/block{block_id}/attn/v_proj",
            f"encoder/block{block_id}/attn/softmax",
            f"encoder/block{block_id}/attn/out_proj",
            f"encoder/block{block_id}/attn/dropout",
            f"encoder/block{block_id}/resid_add1",
            f"encoder/block{block_id}/norm2",
            f"encoder/block{block_id}/ffn/linear1",
            f"encoder/block{block_id}/ffn/activation",
            f"encoder/block{block_id}/ffn/dropout1",
            f"encoder/block{block_id}/ffn/linear2",
            f"encoder/block{block_id}/ffn/dropout2",
            f"encoder/block{block_id}/resid_add2",
        ]
    
    layer_groups["final"] = ["final/norm", "pooler/mean", "pooler/tanh"]
    layer_groups["classifier"] = [
        "classifier/linear1", "classifier/norm1", "classifier/activation1",
        "classifier/dropout1", "classifier/linear2", "classifier/norm2",
        "classifier/activation2", "classifier/dropout2", "classifier/linear3",
        "classifier/output", "classifier/softmax"
    ]
    
    payload = {
        "metadata": {
            "run_id": run_id,
            "timestamp": time.time(),
            "global_step": global_step,
            "batch_size": batch_size,
            "layer_groups": layer_groups
        },
        "layer_statistics": layer_statistics,
        "cross_layer_analysis": {
            "feature_std_gradient": feature_std_gradient,
            "gradient_norm_ratio": gradient_norm_ratios
        }
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload, timeout=10)
        if response.status_code == 202:
            alert_count = count_brutal_alerts(layer_statistics)
            return True, alert_count
        else:
            return False, 0
    except Exception as e:
        return False, 0


def count_brutal_alerts(layers: List[Dict]) -> int:
    """Count alerts - each layer can trigger multiple alerts"""
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


def brutal_flood(run_id: str, steps: int, interval: float):
    """Flood the server with massive alert data"""
    
    severities = ["extreme", "vanishing", "exploding"]
    total_alerts = 0
    
    print(f"\n{'='*70}")
    print(f"BRUTAL STRESS TEST: {run_id}")
    print(f"{'='*70}")
    print(f"Total layers: {len(BRUTAL_LAYERS)}")
    print(f"Total steps: {steps}")
    print(f"Expected alerts: ~{len(BRUTAL_LAYERS) * 2 * steps}")
    print(f"{'='*70}\n")
    
    for step in range(steps):
        severity = severities[step % len(severities)]
        success, alerts = send_brutal_metrics(run_id, step, severity=severity)
        
        if success:
            total_alerts += alerts
            symbol = "🔥" if alerts > 150 else "⚠️" if alerts > 50 else "📊"
            print(f"{symbol} [Step {step:4d}] Alerts: {alerts:4d} | Total: {total_alerts:6d}")
        else:
            print(f"❌ [Step {step:4d}] FAILED")
        
        time.sleep(interval)
    
    print(f"\n{'='*70}")
    print(f"Test complete! Total alerts generated: {total_alerts}")
    print(f"{'='*70}\n")
    return total_alerts


def multi_run_assault(num_runs: int = 3, steps_per_run: int = 50):
    """Launch multiple runs simultaneously for maximum stress"""
    
    print(f"\n{'#'*70}")
    print(f"MULTI-RUN ASSAULT: {num_runs} runs × {steps_per_run} steps")
    print(f"{'#'*70}\n")
    
    def run_test(run_idx):
        run_id = f"BRUTAL_{run_idx}_{int(time.time())}"
        return brutal_flood(run_id, steps_per_run, interval=0.1)
    
    with ThreadPoolExecutor(max_workers=num_runs) as executor:
        futures = [executor.submit(run_test, i) for i in range(num_runs)]
        results = [f.result() for f in futures]
    
    total = sum(results)
    print(f"\n{'#'*70}")
    print(f"ASSAULT COMPLETE!")
    print(f"Total alerts across all runs: {total}")
    print(f"{'#'*70}\n")


def infinite_flood(run_id: str = "INFINITE_FLOOD"):
    """Never-ending flood (Ctrl+C to stop)"""
    
    print(f"\n{'💀'*35}")
    print(f"INFINITE FLOOD STARTED: {run_id}")
    print(f"Press Ctrl+C to stop")
    print(f"{'💀'*35}\n")
    
    step = 0
    total_alerts = 0
    
    try:
        while True:
            success, alerts = send_brutal_metrics(run_id, step, severity="extreme")
            if success:
                total_alerts += alerts
                print(f"🔥🔥🔥 [Step {step:5d}] +{alerts:4d} alerts | Total: {total_alerts:7d}")
            step += 1
            time.sleep(0.05)  # 20 updates per second
    except KeyboardInterrupt:
        print(f"\n\nStopped after {step} steps and {total_alerts} total alerts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BRUTAL stress test for NN Monitor frontend"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "infinite"],
        default="single",
        help="Test mode: single run, multi-run assault, or infinite flood"
    )
    parser.add_argument(
        "--run-id",
        default=f"BRUTAL_{int(time.time())}",
        help="Run identifier"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps (single mode)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of parallel runs (multi mode)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between updates"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        brutal_flood(args.run_id, args.steps, args.interval)
    elif args.mode == "multi":
        multi_run_assault(args.runs, args.steps)
    elif args.mode == "infinite":
        infinite_flood(args.run_id)
