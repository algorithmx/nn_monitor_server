"""
Comprehensive test client for NN Training Monitor Server
Generates extensive warnings and alerts to test sidebar scrolling
"""

import requests
import time
import random
import argparse
from typing import List, Dict, Any, Optional

API_ENDPOINT = "http://localhost:8000/api/v1/metrics/layerwise"

# Deep network architecture to generate more alerts
DEEP_LAYERS = [
    ("input/embedding", "Embedding"),
    ("encoder/block1/attn/query", "Linear"),
    ("encoder/block1/attn/key", "Linear"),
    ("encoder/block1/attn/value", "Linear"),
    ("encoder/block1/attn/softmax", "Softmax"),
    ("encoder/block1/norm1", "LayerNorm"),
    ("encoder/block1/ffn/linear1", "Linear"),
    ("encoder/block1/ffn/relu", "ReLU"),
    ("encoder/block1/ffn/linear2", "Linear"),
    ("encoder/block1/norm2", "LayerNorm"),
    ("encoder/block2/attn/query", "Linear"),
    ("encoder/block2/attn/key", "Linear"),
    ("encoder/block2/attn/value", "Linear"),
    ("encoder/block2/attn/softmax", "Softmax"),
    ("encoder/block2/norm1", "LayerNorm"),
    ("encoder/block2/ffn/linear1", "Linear"),
    ("encoder/block2/ffn/relu", "ReLU"),
    ("encoder/block2/ffn/dropout", "Dropout"),
    ("encoder/block2/ffn/linear2", "Linear"),
    ("encoder/block2/norm2", "LayerNorm"),
    ("encoder/block3/attn/query", "Linear"),
    ("encoder/block3/attn/key", "Linear"),
    ("encoder/block3/attn/value", "Linear"),
    ("encoder/block3/attn/softmax", "Softmax"),
    ("encoder/block3/norm1", "LayerNorm"),
    ("encoder/block3/ffn/linear1", "Linear"),
    ("encoder/block3/ffn/gelu", "GELU"),
    ("encoder/block3/ffn/linear2", "Linear"),
    ("encoder/block3/norm2", "LayerNorm"),
    ("pooler/avg", "AdaptiveAvgPool1d"),
    ("classifier/linear1", "Linear"),
    ("classifier/dropout", "Dropout"),
    ("classifier/linear2", "Linear"),
    ("classifier/output", "Linear"),
]


def generate_problematic_layer_stats(
    layer_id: str, 
    layer_type: str, 
    depth_index: int,
    prev_std: Optional[float] = None,
    problem_type: str = "random"  # "random", "vanishing", "exploding", "dead_neurons"
) -> Dict[str, Any]:
    """Generate layer statistics with intentional problems to trigger alerts"""
    
    # Base values
    activation_std = random.uniform(0.3, 1.0)
    gradient_l2_norm = random.uniform(0.01, 0.5)
    gradient_max_abs = gradient_l2_norm * random.uniform(1.5, 3.0)
    
    # Apply problem scenarios based on type and randomness
    if problem_type == "vanishing" or (problem_type == "random" and random.random() < 0.25):
        # Vanishing gradients - triggers alert when < 0.001
        if random.random() < 0.5:
            gradient_l2_norm = random.uniform(0.0001, 0.0009)
            gradient_max_abs = gradient_l2_norm * random.uniform(1.0, 2.0)
    
    if problem_type == "exploding" or (problem_type == "random" and random.random() < 0.15):
        # Exploding gradients - triggers alert when > 10
        gradient_l2_norm = random.uniform(5.0, 15.0)
        gradient_max_abs = random.uniform(15.0, 50.0)  # Definitely > 10
    
    if problem_type == "dead_neurons":
        # Dead neurons - very low activation
        activation_std = random.uniform(0.001, 0.01)
    
    # Severe activation drop - triggers alert when ratio < 0.1
    cross_layer_ratio = None
    if prev_std is not None:
        if random.random() < 0.2:  # 20% chance of severe drop
            cross_layer_ratio = random.uniform(0.01, 0.09)  # Below 0.1 threshold
        else:
            cross_layer_ratio = activation_std / (prev_std + 1e-8)
    
    return {
        "layer_id": layer_id,
        "layer_type": layer_type,
        "depth_index": depth_index,
        "intermediate_features": {
            "activation_std": activation_std,
            "activation_mean": random.uniform(-0.1, 0.1),
            "activation_shape": [random.randint(32, 128), random.randint(64, 512)],
            "cross_layer_std_ratio": cross_layer_ratio
        },
        "gradient_flow": {
            "gradient_l2_norm": gradient_l2_norm,
            "gradient_std": gradient_l2_norm * random.uniform(0.01, 0.1),
            "gradient_max_abs": gradient_max_abs
        },
        "parameter_statistics": {
            "weight": {
                "std": random.uniform(0.01, 0.1),
                "mean": random.uniform(-0.01, 0.01),
                "spectral_norm": random.uniform(0.8, 1.5),
                "frobenius_norm": random.uniform(1.0, 5.0)
            },
            "bias": {
                "std": random.uniform(0.001, 0.01),
                "mean_abs": random.uniform(0.001, 0.01)
            }
        }
    }


def send_metrics(
    run_id: str, 
    global_step: int, 
    batch_size: int = 64,
    problem_scenario: str = "random"
) -> bool:
    """Send metrics payload with intentional problems to trigger alerts"""
    
    layer_statistics = []
    prev_std = None
    prev_grad_norm = None
    gradient_norm_ratios = {}
    
    for depth_idx, (layer_id, layer_type) in enumerate(DEEP_LAYERS):
        layer_data = generate_problematic_layer_stats(
            layer_id, layer_type, depth_idx, prev_std, problem_scenario
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
    feature_std_gradient = (stds[-1] - stds[0]) / len(stds) if len(stds) > 1 else 0.0
    
    # Define layer groups
    layer_groups = {
        "input": ["input/embedding"],
        "encoder/block1": [
            "encoder/block1/attn/query", "encoder/block1/attn/key", 
            "encoder/block1/attn/value", "encoder/block1/attn/softmax",
            "encoder/block1/norm1", "encoder/block1/ffn/linear1",
            "encoder/block1/ffn/relu", "encoder/block1/ffn/linear2",
            "encoder/block1/norm2"
        ],
        "encoder/block2": [
            "encoder/block2/attn/query", "encoder/block2/attn/key",
            "encoder/block2/attn/value", "encoder/block2/attn/softmax",
            "encoder/block2/norm1", "encoder/block2/ffn/linear1",
            "encoder/block2/ffn/relu", "encoder/block2/ffn/dropout",
            "encoder/block2/ffn/linear2", "encoder/block2/norm2"
        ],
        "encoder/block3": [
            "encoder/block3/attn/query", "encoder/block3/attn/key",
            "encoder/block3/attn/value", "encoder/block3/attn/softmax",
            "encoder/block3/norm1", "encoder/block3/ffn/linear1",
            "encoder/block3/ffn/gelu", "encoder/block3/ffn/linear2",
            "encoder/block3/norm2"
        ],
        "pooler": ["pooler/avg"],
        "classifier": [
            "classifier/linear1", "classifier/dropout",
            "classifier/linear2", "classifier/output"
        ]
    }
    
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
        response = requests.post(API_ENDPOINT, json=payload, timeout=5)
        if response.status_code == 202:
            alert_count = count_alerts(layer_statistics)
            print(f"[Step {global_step:4d}] Sent | Alerts: {alert_count}")
            return True
        else:
            print(f"[Step {global_step:4d}] Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"[Step {global_step:4d}] Failed: {e}")
        return False


def count_alerts(layers: List[Dict]) -> int:
    """Count how many alerts this step will generate"""
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


def simulate_problematic_training(
    run_id: str = "test_run", 
    num_steps: int = 50,
    interval: float = 0.3
):
    """Simulate training with various problematic scenarios"""
    
    scenarios = ["random", "vanishing", "exploding", "dead_neurons"]
    
    print(f"=" * 60)
    print(f"COMPREHENSIVE MOCK TEST: {run_id}")
    print(f"=" * 60)
    print(f"Steps: {num_steps}")
    print(f"Layers: {len(DEEP_LAYERS)} (deep network)")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"=" * 60)
    
    for step in range(num_steps):
        # Cycle through different problem scenarios
        scenario = scenarios[step % len(scenarios)]
        send_metrics(run_id, step, problem_scenario=scenario)
        time.sleep(interval)
    
    print("=" * 60)
    print("Test complete! Check the UI for alerts in sidebar.")
    print("=" * 60)


def run_multiple_tests():
    """Run multiple test scenarios to generate many alerts"""
    
    test_runs = [
        ("vanishing_gradients_test", 30, "vanishing"),
        ("exploding_gradients_test", 30, "exploding"),
        ("mixed_problems_test", 50, "random"),
        ("deep_network_test", 40, "random"),
    ]
    
    for run_name, steps, scenario in test_runs:
        print(f"\n>>> Starting run: {run_name}")
        simulate_problematic_training(run_name, steps, interval=0.2)
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive test client with alert generation"
    )
    parser.add_argument(
        "--run-id", 
        default=f"alert_test_{int(time.time())}",
        help="Run identifier"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=100,
        help="Number of training steps"
    )
    parser.add_argument(
        "--interval", 
        type=float, 
        default=0.2,
        help="Seconds between updates"
    )
    parser.add_argument(
        "--endpoint", 
        default=API_ENDPOINT,
        help="Server endpoint URL"
    )
    parser.add_argument(
        "--multi-run", 
        action="store_true",
        help="Run multiple test scenarios"
    )
    
    args = parser.parse_args()
    API_ENDPOINT = args.endpoint
    
    if args.multi_run:
        run_multiple_tests()
    else:
        simulate_problematic_training(args.run_id, args.steps, args.interval)
