"""
Test client for NN Training Monitor Server
Simulates a training monitor sending metrics to the server
"""

import requests
import time
import random
import numpy as np
from typing import List, Dict, Any

API_ENDPOINT = "http://localhost:8000/api/v1/metrics/layerwise"


def generate_layer_stats(layer_id: str, layer_type: str, depth_index: int, 
                         prev_std: float = None) -> Dict[str, Any]:
    """Generate realistic layer statistics"""
    
    # Simulate activation std with some randomness
    activation_std = random.uniform(0.3, 1.2)
    if prev_std and random.random() < 0.1:  # 10% chance of vanishing
        activation_std = prev_std * random.uniform(0.1, 0.3)
    
    cross_layer_ratio = None
    if prev_std is not None:
        cross_layer_ratio = activation_std / (prev_std + 1e-8)
    
    # Simulate gradient flow (typically smaller in deeper layers)
    base_grad = 0.5 / (depth_index + 1)
    gradient_l2_norm = base_grad * random.uniform(0.5, 2.0)
    
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
            "gradient_max_abs": gradient_l2_norm * random.uniform(1.5, 5.0)
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


def send_metrics(run_id: str, global_step: int, batch_size: int = 64) -> bool:
    """Send metrics payload to server"""
    
    # Define model layers
    layers_config = [
        ("encoder.linear1", "Linear"),
        ("encoder.relu1", "ReLU"),
        ("encoder.linear2", "Linear"),
        ("encoder.relu2", "ReLU"),
        ("classifier", "Linear")
    ]
    
    layer_statistics = []
    prev_std = None
    prev_grad_norm = None
    gradient_norm_ratios = {}
    
    for depth_idx, (layer_id, layer_type) in enumerate(layers_config):
        layer_data = generate_layer_stats(layer_id, layer_type, depth_idx, prev_std)
        layer_statistics.append(layer_data)
        
        current_std = layer_data["intermediate_features"]["activation_std"]
        current_grad_norm = layer_data["gradient_flow"]["gradient_l2_norm"]
        
        # Compute gradient norm ratio
        if prev_grad_norm is not None and prev_grad_norm > 0:
            ratio = current_grad_norm / prev_grad_norm
            gradient_norm_ratios[f"{layer_id}_to_prev"] = ratio
        
        prev_std = current_std
        prev_grad_norm = current_grad_norm
    
    # Compute feature std gradient (trend across depth)
    stds = [l["intermediate_features"]["activation_std"] for l in layer_statistics]
    feature_std_gradient = (stds[-1] - stds[0]) / len(stds) if len(stds) > 1 else 0.0
    
    payload = {
        "metadata": {
            "run_id": run_id,
            "timestamp": time.time(),
            "global_step": global_step,
            "batch_size": batch_size
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
            print(f"[Step {global_step}] Metrics sent successfully")
            return True
        else:
            print(f"[Step {global_step}] Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[Step {global_step}] Failed to send: {e}")
        return False


def simulate_training(run_id: str = "test_run", num_steps: int = 100, 
                      interval: float = 1.0):
    """Simulate a training loop sending metrics"""
    
    print(f"Starting simulated training: {run_id}")
    print(f"Steps: {num_steps}, Interval: {interval}s")
    print(f"Target: {API_ENDPOINT}")
    print("-" * 50)
    
    for step in range(0, num_steps, 10):  # Log every 10 steps
        send_metrics(run_id, step)
        time.sleep(interval)
    
    print("-" * 50)
    print("Training simulation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test client for NN Monitor Server")
    parser.add_argument("--run-id", default=f"test_run_{int(time.time())}", 
                       help="Run identifier")
    parser.add_argument("--steps", type=int, default=100, 
                       help="Number of training steps")
    parser.add_argument("--interval", type=float, default=1.0, 
                       help="Seconds between updates")
    parser.add_argument("--endpoint", default=API_ENDPOINT,
                       help="Server endpoint URL")
    
    args = parser.parse_args()
    
    API_ENDPOINT = args.endpoint
    simulate_training(args.run_id, args.steps, args.interval)
