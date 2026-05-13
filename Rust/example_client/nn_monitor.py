"""
Neural Network Training Monitor
Async HTTP client for posting layer-wise metrics to monitoring server.
"""

import torch
import torch.nn as nn
import asyncio
import threading
import queue
import time
import json
import math
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from urllib.request import Request, urlopen
from urllib.error import URLError


class AsyncMetricsPoster:
    """
    Async HTTP client that batches and POSTs metrics in background thread.
    Non-blocking from training loop perspective. Uses threading to avoid
    requiring external async libraries like aiohttp.
    """
    def __init__(self, api_endpoint: str, run_id: str,
                 batch_size: int = 50, flush_interval: float = 2.0,
                 max_queue_size: int = 2000):
        self.api_endpoint = api_endpoint
        self.run_id = run_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Thread-safe queue for passing metrics from main thread to worker
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background worker: batches metrics and POSTs them"""
        batch = []
        last_flush = time.time()

        while not self._shutdown:
            try:
                # Non-blocking check with timeout for graceful shutdown
                item = self._queue.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                time.sleep(0.01)

            # Flush if batch full or interval passed
            if len(batch) >= self.batch_size or (batch and time.time() - last_flush > self.flush_interval):
                self._post_batch(batch)
                batch = []
                last_flush = time.time()

        # Final flush on shutdown
        if batch:
            self._post_batch(batch)

    def _post_batch(self, batch: list):
        """Send batch of metrics to server using urllib (no aiohttp dependency)"""
        if not batch:
            return

        # Group by step for efficient transmission
        steps = defaultdict(dict)
        for item in batch:
            steps[item['step']][item['tag']] = item['value']

        payload = {
            "run_id": self.run_id,
            "timestamp": time.time(),
            "batches": [
                {"step": step, "metrics": metrics}
                for step, metrics in sorted(steps.items())
            ]
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = Request(
                self.api_endpoint,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urlopen(req, timeout=10) as resp:
                if resp.status >= 400:
                    print(f"[Monitor] Server error {resp.status}")
        except Exception as e:
            # Fail silently to not interrupt training
            pass

    def log_scalar(self, step: int, tag: str, value: float):
        """
        Thread-safe method called from training loop.
        Drops oldest items if queue full (prioritizes training speed).
        """
        item = {
            "step": step,
            "tag": tag,
            "value": float(value),
            "timestamp": time.time()
        }

        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop oldest to make room
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except queue.Empty:
                pass

    def post_payload(self, payload: dict):
        """Post a complete structured payload directly"""
        try:
            data = json.dumps(payload).encode('utf-8')
            req = Request(
                self.api_endpoint,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urlopen(req, timeout=10) as resp:
                if resp.status >= 400:
                    print(f"[Monitor] Server error {resp.status}")
        except Exception as e:
            # Fail silently to not interrupt training
            pass

    def close(self):
        """Graceful shutdown"""
        self._shutdown = True
        self._thread.join(timeout=10.0)


class LayerwiseMonitor:
    """
    Monitors layer-wise statistics and POSTs async to custom JSON API.

    Captures:
    - Intermediate features: activation std/mean between layers, cross-layer ratios
    - Gradient flow: L2 norms of gradients per layer
    - Parameter statistics: std/norms of weights and biases per layer
    """

    def __init__(self, model: nn.Module, api_endpoint: str, run_id: str = None,
                 log_interval: int = 10, batch_size: int = 64, layer_groups: Dict[str, List[str]] = None):
        self.model = model
        self.api_endpoint = api_endpoint
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.global_step = 0
        # Sanitize layer group IDs to match sanitized layer IDs (converts dots to slashes)
        self.layer_groups = self._sanitize_layer_groups(layer_groups) if layer_groups else None

        # Generate run_id if not provided
        if run_id is None:
            run_id = f"fashion_mnist_{time.strftime('%Y%m%d_%H%M%S')}"

        self.run_id = run_id

        # Hook storage
        self.hooks = []
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        self._monitored_layers: List[Tuple[str, nn.Module]] = []

        self._register_hooks()

        print(f"[Monitor] Initialized with run_id: {run_id}")
        print(f"[Monitor] Endpoint: {api_endpoint}")
        print(f"[Monitor] Logging every {log_interval} steps")

    def _register_hooks(self):
        """Register forward/backward hooks on all parameter-bearing layers.

        Skips the root model (empty name) and only includes leaf modules
        that have their own parameters (not just child modules).
        """
        for name, module in self.model.named_modules():
            # Skip root model (empty name)
            if not name:
                continue

            # Only include modules with their own parameters
            # (not just children with parameters)
            has_own_params = False
            for param_name in ['weight', 'bias']:
                if hasattr(module, param_name):
                    has_own_params = True
                    break

            if has_own_params:
                self._monitored_layers.append((name, module))

                # Forward: capture activations
                handle_fwd = module.register_forward_hook(
                    self._make_forward_hook(name)
                )
                self.hooks.append(handle_fwd)

                # Backward: capture gradients
                handle_bwd = module.register_full_backward_hook(
                    self._make_backward_hook(name)
                )
                self.hooks.append(handle_bwd)

        print(f"[Monitor] Registered hooks for {len(self._monitored_layers)} layers")

    def _make_forward_hook(self, name: str):
        """Create a forward hook for capturing activations"""
        def hook(module, inp, out):
            if self.global_step % self.log_interval == 0 and out is not None:
                self._activations[name] = out.detach().clone()
        return hook

    def _make_backward_hook(self, name: str):
        """Create a backward hook for capturing gradients"""
        def hook(module, grad_in, grad_out):
            if self.global_step % self.log_interval == 0:
                if grad_out and grad_out[0] is not None:
                    self._gradients[name] = grad_out[0].detach().clone()
        return hook

    def _sanitize_name(self, name: str) -> str:
        """Clean names for JSON keys"""
        return name.replace('.', '/')

    def _sanitize_layer_groups(self, layer_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Sanitize layer group IDs to match sanitized layer IDs (converts dots to slashes).

        This ensures that layer group definitions match the sanitized layer IDs
        sent to the server, even if the user provides layer IDs with dots.

        Args:
            layer_groups: Dict mapping group names to lists of layer IDs

        Returns:
            Dict with all layer IDs sanitized (dots replaced with slashes)
        """
        if not layer_groups:
            return layer_groups
        return {
            group_name: [self._sanitize_name(layer_id) for layer_id in layer_ids]
            for group_name, layer_ids in layer_groups.items()
        }

    def _approx_spectral_norm(self, W: torch.Tensor, n_iter: int = 3) -> float:
        """Power iteration for spectral norm approximation.

        Handles 1D, 2D (Linear), and 4D (Conv2d) tensors.
        For Conv2d, we reshape to 2D (out_channels, in_channels * kernel_h * kernel_w).
        """
        if len(W.shape) < 2:
            return torch.norm(W).item()

        device = W.device

        # Reshape 4D tensors (Conv2d weights) to 2D for spectral norm
        if len(W.shape) == 4:
            # W.shape = [out_channels, in_channels, kernel_h, kernel_w]
            W_2d = W.reshape(W.shape[0], -1)  # [out_channels, in_channels * kernel_h * kernel_w]
        else:
            W_2d = W

        # Power iteration for spectral norm
        u = torch.randn(W_2d.shape[0], 1, device=device)

        for _ in range(n_iter):
            v = (W_2d.t() @ u) / (torch.norm(W_2d.t() @ u) + 1e-8)
            u = (W_2d @ v) / (torch.norm(W_2d @ v) + 1e-8)

        return (u.t() @ W_2d @ v).item()

    def _is_finite(self, val: float) -> bool:
        """Check if value is finite (not NaN or Infinity)"""
        return math.isfinite(val)

    def _build_layer_statistics(self) -> List[Dict[str, Any]]:
        """Build layer statistics list for the API payload"""
        layer_stats = []

        for depth_idx, (name, module) in enumerate(self._monitored_layers):
            layer_type = module.__class__.__name__
            clean_name = self._sanitize_name(name)

            layer_data = {
                "layer_id": clean_name,
                "layer_type": layer_type,
                "depth_index": depth_idx,
                "intermediate_features": None,
                "gradient_flow": None,
                "parameter_statistics": {
                    "weight": None,
                    "bias": None
                }
            }

            # 1. Intermediate features (activations)
            if name in self._activations:
                act = self._activations[name]
                act_std = act.std().item()
                act_mean = act.mean().item()
                act_shape = list(act.shape)

                if all(self._is_finite(v) for v in [act_std, act_mean]):
                    layer_data["intermediate_features"] = {
                        "activation_std": act_std,
                        "activation_mean": act_mean,
                        "activation_shape": act_shape,
                        "cross_layer_std_ratio": None
                    }

            # 2. Gradient flow
            if name in self._gradients:
                grad = self._gradients[name]
                grad_norm = torch.norm(grad).item()
                grad_std = grad.std().item()
                grad_max_abs = grad.abs().max().item()

                if all(self._is_finite(v) for v in [grad_norm, grad_std, grad_max_abs]):
                    layer_data["gradient_flow"] = {
                        "gradient_l2_norm": grad_norm,
                        "gradient_std": grad_std,
                        "gradient_max_abs": grad_max_abs
                    }

            # 3. Parameter statistics (weights and biases)
            for param_name, param in module.named_parameters():
                if not param.requires_grad:
                    continue

                param_std = param.std().item()
                param_mean = param.mean().item()

                if not all(self._is_finite(v) for v in [param_std, param_mean]):
                    continue

                if "weight" in param_name:
                    spectral_norm = self._approx_spectral_norm(param)
                    frobenius_norm = torch.norm(param).item()

                    if self._is_finite(spectral_norm) and self._is_finite(frobenius_norm):
                        layer_data["parameter_statistics"]["weight"] = {
                            "std": param_std,
                            "mean": param_mean,
                            "spectral_norm": spectral_norm,
                            "frobenius_norm": frobenius_norm
                        }

                elif "bias" in param_name:
                    mean_abs = param.abs().mean().item()

                    if self._is_finite(mean_abs):
                        layer_data["parameter_statistics"]["bias"] = {
                            "std": param_std,
                            "mean_abs": mean_abs
                        }

            # Only add if we have at least one non-None section
            if (layer_data["intermediate_features"] is not None or
                layer_data["gradient_flow"] is not None or
                layer_data["parameter_statistics"]["weight"] is not None or
                layer_data["parameter_statistics"]["bias"] is not None):
                layer_stats.append(layer_data)

        return layer_stats

    def _compute_cross_layer_analysis(self, layer_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute cross-layer analysis metrics"""
        analysis = {
            "feature_std_gradient": 0.0,
            "gradient_norm_ratio": {}
        }

        # Compute feature std gradient (trend across depth)
        stds = []
        for layer in layer_stats:
            if layer["intermediate_features"] is not None:
                stds.append(layer["intermediate_features"]["activation_std"])

        if len(stds) > 1:
            # Linear regression slope approximation
            analysis["feature_std_gradient"] = (stds[-1] - stds[0]) / len(stds)

        # Compute gradient norm ratios between consecutive layers
        prev_grad_norm = None
        for layer in layer_stats:
            if layer["gradient_flow"] is not None:
                curr_norm = layer["gradient_flow"]["gradient_l2_norm"]
                if prev_grad_norm is not None and prev_grad_norm > 0:
                    ratio_key = f"{layer['layer_id']}_to_prev"
                    analysis["gradient_norm_ratio"][ratio_key] = curr_norm / prev_grad_norm
                prev_grad_norm = curr_norm

        # Add cross_layer_std_ratio to intermediate_features
        prev_std = None
        for layer in layer_stats:
            if layer["intermediate_features"] is not None:
                curr_std = layer["intermediate_features"]["activation_std"]
                if prev_std is not None and prev_std > 0:
                    layer["intermediate_features"]["cross_layer_std_ratio"] = curr_std / prev_std
                prev_std = curr_std

        return analysis

    def log_metrics(self):
        """Collect and send metrics to the monitoring server"""
        if self.global_step % self.log_interval != 0:
            return

        # Build the payload structure according to the API schema
        layer_stats = self._build_layer_statistics()

        if not layer_stats:
            return

        cross_layer = self._compute_cross_layer_analysis(layer_stats)

        payload = {
            "metadata": {
                "run_id": self.run_id,
                "timestamp": time.time(),
                "global_step": self.global_step,
                "batch_size": self.batch_size,
                "layer_groups": self.layer_groups
            },
            "layer_statistics": layer_stats,
            "cross_layer_analysis": cross_layer
        }

        # Send to server (blocking but simple)
        try:
            data = json.dumps(payload).encode('utf-8')
            req = Request(
                self.api_endpoint,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urlopen(req, timeout=5) as resp:
                if resp.status < 400:
                    print(f"[Monitor] Step {self.global_step}: Metrics posted successfully")
                else:
                    # Read error response
                    error_body = resp.read().decode('utf-8')
                    print(f"[Monitor] Step {self.global_step}: Server returned {resp.status} - {error_body[:200]}")
        except URLError as e:
            print(f"[Monitor] Step {self.global_step}: Could not connect to server - {e}")
        except Exception as e:
            print(f"[Monitor] Step {self.global_step}: Error posting metrics - {e}")

        # Clear GPU memory buffers
        self._activations.clear()
        self._gradients.clear()

    def step(self):
        """Call at end of each training step.

        Increments the step counter. The actual logging should be done
        by calling log_metrics() directly at the appropriate point in
        the training loop (typically after backward() but before optimizer.step()).
        """
        self.global_step += 1

    def close(self):
        """Cleanup hooks"""
        for hook in self.hooks:
            hook.remove()
        print(f"[Monitor] Closed. Total steps: {self.global_step}")


def create_monitor(model: nn.Module, api_endpoint: str = "http://localhost:8000/api/v1/metrics/layerwise",
                   run_id: str = None, log_interval: int = 10, batch_size: int = 64,
                   layer_groups: Dict[str, List[str]] = None) -> LayerwiseMonitor:
    """
    Convenience function to create and return a LayerwiseMonitor.

    Args:
        model: PyTorch model to monitor
        api_endpoint: URL of the monitoring server API endpoint
        run_id: Optional unique identifier for this training run
        log_interval: Post metrics every N steps
        batch_size: Batch size for training metadata
        layer_groups: Optional dict mapping group names to lists of layer_ids

    Returns:
        LayerwiseMonitor instance
    """
    return LayerwiseMonitor(
        model=model,
        api_endpoint=api_endpoint,
        run_id=run_id,
        log_interval=log_interval,
        batch_size=batch_size,
        layer_groups=layer_groups
    )
