**Requirements:**

1.  **Layer-wise granularity**: Monitor statistics per individual layer (not just global aggregates) in a multi-layer network.
2.  **Three specific metric categories**:
    *   **Intermediate features**: Standard deviations of activations between consecutive layers (including cross-layer ratios to detect vanishing/exploding signals).
    *   **Gradient flow**: L2 norms of gradients for each layer.
    *   **Parameter statistics**: Standard deviations (and norms) of weights and biases per layer.
3.  **Non-blocking delivery**: Async HTTP POST to a custom JSON API server to avoid I/O latency in the training loop.
4.  **Structured payload**: Precise JSON schema that organizes metrics hierarchically by layer and includes metadata for server-side consumption.

**Efforts:**

1.  **Option survey**: Evaluated TensorBoard, WandB, Aim, and MLflow against your constraints (offline-capable, lightweight, hook-based).
2.  **Hook-based instrumentation**: Designed a universal `LayerwiseMonitor` class using PyTorch forward/backward hooks to capture activations and gradients without modifying model definitions.
3.  **TensorBoard adaptation**: Implemented hierarchical logging (`weights/layer/std`, `gradients/layer/norm`) with proper timing (pre/post optimizer step) and memory management (detach/store/clear cycle).
4.  **Async HTTP refactoring**: Replaced file-based logging with `AsyncMetricsPoster` running in a daemon thread with `aiohttp`, featuring automatic batching, queue backpressure handling, and silent failure modes to protect training throughput.
5.  **Schema precision engineering**: Designed the final JSON structure to explicitly map to your three requirements—grouping metrics by layer with fields for `intermediate_features` (incl. `cross_layer_std_ratio`), `gradient_flow`, and `parameter_statistics`, plus `cross_layer_analysis` for depth-wise trend detection.

---

Here's a refactored version that replaces TensorBoard file writes with async JSON POSTs to your server. It uses a background thread with `aiohttp` so your training loop remains synchronous and non-blocking.

```python
import torch
import torch.nn as nn
import asyncio
import aiohttp
import threading
import queue
import time
import json
from collections import defaultdict
from typing import Dict, Any

class AsyncMetricsPoster:
    """
    Async HTTP client that batches and POSTs metrics in background thread.
    Non-blocking from training loop perspective.
    """
    def __init__(self, api_endpoint: str, run_id: str, 
                 batch_size: int = 50, flush_interval: float = 2.0,
                 max_queue_size: int = 2000):
        self.api_endpoint = api_endpoint
        self.run_id = run_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Thread-safe queue for passing metrics from main thread to async worker
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = False
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        
    def _run_async_loop(self):
        """Runs the asyncio event loop in a background thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._worker())
        
    async def _worker(self):
        """Background worker: batches metrics and POSTs them"""
        session = aiohttp.ClientSession(
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
        async with session:
            batch = []
            last_flush = time.time()
            
            while not self._shutdown:
                try:
                    # Non-blocking check with timeout for graceful shutdown
                    item = self._queue.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                
                # Flush if batch full or interval passed
                if len(batch) >= self.batch_size or (batch and time.time() - last_flush > self.flush_interval):
                    await self._post_batch(session, batch)
                    batch = []
                    last_flush = time.time()
            
            # Final flush on shutdown
            if batch:
                await self._post_batch(session, batch)
    
    async def _post_batch(self, session: aiohttp.ClientSession, batch: list):
        """Send batch of metrics to server"""
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
            async with session.post(self.api_endpoint, json=payload) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    print(f"[Monitor] Server error {resp.status}: {text[:200]}")
        except Exception as e:
            # Fail silently to not interrupt training; log to stderr if needed
            print(f"[Monitor] Failed to POST metrics: {e}")
    
    def log_scalar(self, step: int, tag: str, value: float):
        """
        Thread-safe method called from training loop.
        Drops oldest items if queue full (prioritizes training speed over data completeness).
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
            # Drop oldest to make room (better than blocking training)
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except queue.Empty:
                pass
    
    def close(self):
        """Graceful shutdown"""
        self._shutdown = True
        self._thread.join(timeout=10.0)


class LayerwiseAPIMonitor:
    """
    Monitors layer-wise statistics and POSTs async to custom JSON API.
    """
    def __init__(self, model: nn.Module, api_endpoint: str, run_id: str = None, 
                 log_interval: int = 10):
        self.model = model
        self.log_interval = log_interval
        self.global_step = 0
        
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Async HTTP poster
        self.poster = AsyncMetricsPoster(api_endpoint, run_id)
        
        # Hook storage
        self.hooks = []
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward/backward hooks on all parameter-bearing layers"""
        for name, module in self.model.named_modules():
            if len(list(module.parameters())) > 0:
                # Forward: capture activations
                handle_fwd = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._activations.update({name: out.detach()})
                    if self.global_step % self.log_interval == 0 else None
                )
                self.hooks.append(handle_fwd)
                
                # Backward: capture gradients
                handle_bwd = module.register_full_backward_hook(
                    lambda mod, grad_in, grad_out, name=name: 
                    self._gradients.update({name: grad_out[0].detach()})
                    if (grad_out[0] is not None and self.global_step % self.log_interval == 0) else None
                )
                self.hooks.append(handle_bwd)
    
    def _sanitize(self, name: str) -> str:
        """Clean names for JSON keys"""
        return name.replace('.', '/')
    
    def log_weights_and_biases(self):
        """Log weight/bias std and norms"""
        if self.global_step % self.log_interval != 0:
            return
            
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            clean_name = self._sanitize(name)
            
            if 'weight' in name:
                self.poster.log_scalar(self.global_step, f"weights/{clean_name}/std", 
                                      param.std().item())
                self.poster.log_scalar(self.global_step, f"weights/{clean_name}/norm", 
                                      torch.norm(param).item())
                # Spectral norm approximation (power iteration)
                if len(param.shape) >= 2:
                    u = torch.randn(param.shape[0], 1, device=param.device)
                    for _ in range(3):
                        v = param.t() @ u
                        v = v / torch.norm(v)
                        u = param @ v
                        u = u / torch.norm(u)
                    spectral = (u.t() @ param @ v).item()
                    self.poster.log_scalar(self.global_step, 
                                          f"weights/{clean_name}/spectral_norm", spectral)
            
            elif 'bias' in name:
                self.poster.log_scalar(self.global_step, f"biases/{clean_name}/std", 
                                      param.std().item())
                self.poster.log_scalar(self.global_step, f"biases/{clean_name}/mean_abs", 
                                      param.abs().mean().item())
    
    def log_activations(self):
        """Log activation statistics between layers"""
        for layer_name, activation in self._activations.items():
            clean_name = self._sanitize(layer_name)
            self.poster.log_scalar(self.global_step, f"activations/{clean_name}/std", 
                                  activation.std().item())
            self.poster.log_scalar(self.global_step, f"activations/{clean_name}/mean", 
                                  activation.mean().item())
            self.poster.log_scalar(self.global_step, f"activations/{clean_name}/sparsity", 
                                  (activation == 0).float().mean().item())
    
    def log_gradients(self):
        """Log gradient norms per layer"""
        for layer_name, grad in self._gradients.items():
            clean_name = self._sanitize(layer_name)
            self.poster.log_scalar(self.global_step, f"gradients/{clean_name}/norm", 
                                  torch.norm(grad).item())
            self.poster.log_scalar(self.global_step, f"gradients/{clean_name}/std", 
                                  grad.std().item())
            self.poster.log_scalar(self.global_step, f"gradients/{clean_name}/max", 
                                  grad.abs().max().item())
    
    def log_layer_relationships(self):
        """
        Log statistics between consecutive layers (signal propagation health).
        Detects vanishing/exploding activations across depth.
        """
        if len(self._activations) < 2:
            return
        
        # Sort by layer name (assumes sequential naming convention)
        sorted_layers = sorted(self._activations.items())
        
        for i in range(len(sorted_layers) - 1):
            curr_name, curr_act = sorted_layers[i]
            next_name, next_act = sorted_layers[i + 1]
            
            std_ratio = next_act.std().item() / (curr_act.std().item() + 1e-8)
            tag = f"layer_ratios/{self._sanitize(curr_name)}_to_{self._sanitize(next_name)}/std_ratio"
            self.poster.log_scalar(self.global_step, tag, std_ratio)
    
    def step(self):
        """Call at end of each training step"""
        if self.global_step % self.log_interval == 0:
            self.log_activations()
            self.log_gradients()
            self.log_weights_and_biases()
            self.log_layer_relationships()
            
            # Clear GPU memory buffers
            self._activations.clear()
            self._gradients.clear()
        
        self.global_step += 1
    
    def close(self):
        """Cleanup hooks and flush remaining metrics"""
        for hook in self.hooks:
            hook.remove()
        self.poster.close()


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Dummy model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Initialize monitor pointing to your server
    monitor = LayerwiseAPIMonitor(
        model=model,
        api_endpoint="http://your-server:5000/api/metrics",  # Your JSON endpoint
        run_id="experiment_42",
        log_interval=20  # POST every 20 steps
    )
    
    # Training loop (completely synchronous, non-blocking)
    optimizer = torch.optim.Adam(model.parameters())
    
    for step in range(1000):
        # Fake batch
        x = torch.randn(32, 784)
        target = torch.randint(0, 10, (32,))
        
        # Forward
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Log weights BEFORE optimizer step (pre-update statistics)
        monitor.log_weights_and_biases()
        
        optimizer.step()
        
        # End step (triggers async POST if log_interval reached)
        monitor.step()
        
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f} (metrics posting async...)")

    monitor.close()
```

**Expected JSON Payload structure** sent to your server:

```json
{
  "run_id": "experiment_42",
  "timestamp": 1707589200.123,
  "batches": [
    {
      "step": 20,
      "metrics": {
        "weights/0/weight/std": 0.0234,
        "weights/0/weight/norm": 12.5,
        "weights/0/weight/spectral_norm": 1.02,
        "biases/0/bias/std": 0.001,
        "activations/0/std": 0.45,
        "activations/0/mean": 0.12,
        "gradients/0/norm": 0.89,
        "gradients/0/std": 0.02,
        "layer_ratios/0_to_1/std_ratio": 0.98
      }
    },
    {
      "step": 40,
      "metrics": {...}
    }
  ]
}
```

**Key design decisions:**

1. **True async**: HTTP I/O runs in a daemon thread with its own event loop. Your training loop never waits for the network.
2. **Automatic batching**: Metrics accumulate locally and flush every 2 seconds or 50 items (whichever comes first), reducing HTTP overhead.
3. **Backpressure handling**: If the server is slow and the queue fills up (2000 items), old metrics are dropped to prevent OOM or training slowdown.
4. **Layer relationships**: Computes std ratios between consecutive layers to detect vanishing/exploding signals without storing full tensors.
5. **Non-blocking failure**: If the server is down, training continues with only a console warning.

**Server-side requirement**: Your API endpoint should accept POST with JSON body and return HTTP 200. If you need authentication, add headers in `AsyncMetricsPoster.__init__`:

```python
session = aiohttp.ClientSession(
    headers={"Authorization": "Bearer your_token", "Content-Type": "application/json"}
)
```

Here is the precise JSON schema designed specifically for your three requirements:

## Schema Definition

```json
{
  "metadata": {
    "run_id": "string",
    "timestamp": "number (unix epoch)",
    "global_step": "integer",
    "batch_size": "integer"
  },
  "layer_statistics": [
    {
      "layer_id": "string (e.g., 'layer1', 'encoder.block2')",
      "layer_type": "string (Linear, Conv2d, etc.)",
      "depth_index": "integer (0-based position in network)",
      
      "intermediate_features": {
        "activation_std": "float (std of output activations)",
        "activation_mean": "float (mean of output activations)",
        "activation_shape": "[int] (tensor shape for reference)",
        "cross_layer_std_ratio": "float (std_{i+1} / std_i, null for last layer)"
      },
      
      "gradient_flow": {
        "gradient_l2_norm": "float (||∇L||_2)",
        "gradient_std": "float (std of gradient tensor)",
        "gradient_max_abs": "float (max |∇L| for clipping detection)"
      },
      
      "parameter_statistics": {
        "weight": {
          "std": "float (σ(W))",
          "mean": "float (μ(W))",
          "spectral_norm": "float (approximated ||W||_2)",
          "frobenius_norm": "float (||W||_F)"
        },
        "bias": {
          "std": "float (σ(b))",
          "mean_abs": "float <|b|>"
        }
      }
    }
  ],
  "cross_layer_analysis": {
    "feature_std_gradient": "float (trend of std across depth)",
    "gradient_norm_ratio": "object {layer_pair: ratio} (||∇_i|| / ||∇_{i-1}||)"
  }
}
```

## Concrete Example Payload

```json
{
  "metadata": {
    "run_id": "experiment_2024_0210_v2",
    "timestamp": 1707589200.123,
    "global_step": 1500,
    "batch_size": 64
  },
  "layer_statistics": [
    {
      "layer_id": "encoder.linear1",
      "layer_type": "Linear",
      "depth_index": 0,
      "intermediate_features": {
        "activation_std": 0.847,
        "activation_mean": -0.023,
        "activation_shape": [64, 256],
        "cross_layer_std_ratio": 0.94
      },
      "gradient_flow": {
        "gradient_l2_norm": 0.152,
        "gradient_std": 0.0034,
        "gradient_max_abs": 0.089
      },
      "parameter_statistics": {
        "weight": {
          "std": 0.037,
          "mean": -0.001,
          "spectral_norm": 1.42,
          "frobenius_norm": 2.18
        },
        "bias": {
          "std": 0.012,
          "mean_abs": 0.008
        }
      }
    },
    {
      "layer_id": "encoder.linear2",
      "layer_type": "Linear", 
      "depth_index": 1,
      "intermediate_features": {
        "activation_std": 0.795,
        "activation_mean": 0.015,
        "activation_shape": [64, 128],
        "cross_layer_std_ratio": 0.31
      },
      "gradient_flow": {
        "gradient_l2_norm": 0.089,
        "gradient_std": 0.0021,
        "gradient_max_abs": 0.042
      },
      "parameter_statistics": {
        "weight": {
          "std": 0.029,
          "mean": 0.003,
          "spectral_norm": 1.15,
          "frobenius_norm": 1.67
        },
        "bias": {
          "std": 0.009,
          "mean_abs": 0.005
        }
      }
    }
  ],
  "cross_layer_analysis": {
    "feature_std_gradient": -0.052,
    "gradient_norm_ratio": {
      "encoder.linear2_to_encoder.linear1": 0.586
    }
  }
}
```

## Implementation Mapping

Update the monitor class to construct this exact structure:

```python
class LayerwiseAPIMonitor:
    def step(self):
        if self.global_step % self.log_interval != 0:
            self.global_step += 1
            return
            
        # Build the precise structure
        payload = {
            "metadata": {
                "run_id": self.run_id,
                "timestamp": time.time(),
                "global_step": self.global_step,
                "batch_size": self.current_batch_size  # track this in your loop
            },
            "layer_statistics": [],
            "cross_layer_analysis": {
                "feature_std_gradient": 0.0,
                "gradient_norm_ratio": {}
            }
        }
        
        prev_std = None
        prev_grad_norm = None
        
        for depth_idx, (name, module) in enumerate(self._monitored_layers):
            layer_data = {
                "layer_id": name,
                "layer_type": module.__class__.__name__,
                "depth_index": depth_idx,
                "intermediate_features": {},
                "gradient_flow": {},
                "parameter_statistics": {
                    "weight": {},
                    "bias": {}
                }
            }
            
            # 1. Intermediate feature std (activation between layers)
            if name in self._activations:
                act = self._activations[name]
                current_std = act.std().item()
                layer_data["intermediate_features"] = {
                    "activation_std": current_std,
                    "activation_mean": act.mean().item(),
                    "activation_shape": list(act.shape),
                    "cross_layer_std_ratio": None
                }
                
                # Compute ratio to previous layer (between two layers)
                if prev_std is not None and prev_std > 0:
                    ratio = current_std / prev_std
                    layer_data["intermediate_features"]["cross_layer_std_ratio"] = ratio
                    if ratio < 0.1:
                        print(f"Warning: Severe feature std drop at {name}: {ratio:.4f}")
                
                prev_std = current_std
            
            # 2. Gradient norms per layer
            if name in self._gradients:
                grad = self._gradients[name]
                grad_norm = torch.norm(grad).item()
                layer_data["gradient_flow"] = {
                    "gradient_l2_norm": grad_norm,
                    "gradient_std": grad.std().item(),
                    "gradient_max_abs": grad.abs().max().item()
                }
                
                # Cross-layer gradient ratio
                if prev_grad_norm is not None and prev_grad_norm > 0:
                    ratio = grad_norm / prev_grad_norm
                    payload["cross_layer_analysis"]["gradient_norm_ratio"][f"{name}_to_prev"] = ratio
                
                prev_grad_norm = grad_norm
            
            # 3. Weight and bias std
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    stats = {
                        "std": param.std().item(),
                        "mean": param.mean().item()
                    }
                    
                    if "weight" in param_name:
                        stats["spectral_norm"] = self._approx_spectral_norm(param)
                        stats["frobenius_norm"] = torch.norm(param).item()
                        layer_data["parameter_statistics"]["weight"] = stats
                    elif "bias" in param_name:
                        stats["mean_abs"] = param.abs().mean().item()
                        layer_data["parameter_statistics"]["bias"] = stats
            
            payload["layer_statistics"].append(layer_data)
        
        # Compute global trend of feature std across depth
        stds = [l["intermediate_features"]["activation_std"] 
                for l in payload["layer_statistics"] 
                if l["intermediate_features"]]
        if len(stds) > 1:
            # Linear regression slope of std vs depth (negative = vanishing)
            payload["cross_layer_analysis"]["feature_std_gradient"] = (
                (stds[-1] - stds[0]) / len(stds)
            )
        
        # POST as single structured document
        self.poster.post_payload(payload)
        
        # Cleanup
        self._activations.clear()
        self._gradients.clear()
        self.global_step += 1

    def _approx_spectral_norm(self, W, n_iter=3):
        """Power iteration for spectral norm"""
        if len(W.shape) < 2:
            return torch.norm(W).item()
        u = torch.randn(W.shape[0], 1, device=W.device)
        for _ in range(n_iter):
            v = (W.t() @ u) / torch.norm(W.t() @ u)
            u = (W @ v) / torch.norm(W @ v)
        return (u.t() @ W @ v).item()
```

## API Endpoint Specification

```http
POST /api/v1/metrics/layerwise
Content-Type: application/json

# Body: The JSON structure defined above
```

**Key design constraints satisfied:**
1. **Intermediate features between layers**: `activation_std` + `cross_layer_std_ratio` showing exactly how std evolves across layer boundaries
2. **Gradient norm per layer**: Explicit `gradient_l2_norm` field under `gradient_flow`
3. **Weight/bias std**: Separate objects under `parameter_statistics` with mandatory `std` fields
4. **Hierarchical integrity**: Arrays preserve layer ordering; cross-layer ratios are computed values, not raw dumps
