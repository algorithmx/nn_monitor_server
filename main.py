"""
Neural Network Training Monitor Server
FastAPI backend with WebSocket support for real-time visualization
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import json
import math
import os

app = FastAPI(title="NN Training Monitor", version="1.0.0", max_request_size=2_000_000)

# ==================== Data Models ====================

class Metadata(BaseModel):
    run_id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_\.-]+$", description="Unique identifier for the training run")
    timestamp: float = Field(..., gt=0, description="Unix epoch timestamp")
    global_step: int = Field(..., ge=0, description="Training step number")
    batch_size: int = Field(..., gt=0, description="Batch size")

class IntermediateFeatures(BaseModel):
    activation_std: float = Field(..., gt=0, le=1000, description="Standard deviation of activations")
    activation_mean: float = Field(..., description="Mean of activations")
    activation_shape: List[int] = Field(..., min_length=2, description="Shape of activation tensor")
    cross_layer_std_ratio: Optional[float] = Field(None, gt=0, description="Cross-layer standard deviation ratio")

    @field_validator('activation_std', 'activation_mean')
    @classmethod
    def validate_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Value must be finite (not NaN or Infinity)")
        return v

    @field_validator('cross_layer_std_ratio')
    @classmethod
    def validate_finite_optional(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("Value must be finite (not NaN or Infinity)")
        return v

class GradientFlow(BaseModel):
    gradient_l2_norm: float = Field(..., ge=0, le=1e6, description="L2 norm of gradients")
    gradient_std: float = Field(..., ge=0, description="Standard deviation of gradients")
    gradient_max_abs: float = Field(..., ge=0, description="Maximum absolute gradient value")

    @field_validator('gradient_l2_norm', 'gradient_std', 'gradient_max_abs')
    @classmethod
    def validate_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Value must be finite (not NaN or Infinity)")
        return v

class WeightStats(BaseModel):
    std: float = Field(..., gt=0, le=100, description="Standard deviation of weights")
    mean: float = Field(..., description="Mean of weights")
    spectral_norm: float = Field(..., gt=0, le=1e4, description="Spectral norm of weight matrix")
    frobenius_norm: float = Field(..., gt=0, le=1e4, description="Frobenius norm of weight matrix")

    @field_validator('std', 'mean', 'spectral_norm', 'frobenius_norm')
    @classmethod
    def validate_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Value must be finite (not NaN or Infinity)")
        return v

class BiasStats(BaseModel):
    std: float = Field(..., ge=0, le=10, description="Standard deviation of bias")
    mean_abs: float = Field(..., ge=0, description="Mean absolute value of bias")

    @field_validator('std', 'mean_abs')
    @classmethod
    def validate_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Value must be finite (not NaN or Infinity)")
        return v

class ParameterStatistics(BaseModel):
    weight: WeightStats
    bias: BiasStats

class LayerStatistic(BaseModel):
    layer_id: str = Field(..., min_length=1, description="Unique identifier for the layer")
    layer_type: str = Field(..., min_length=1, description="Type of layer (e.g., 'Linear', 'Conv2d')")
    depth_index: int = Field(..., ge=0, description="Depth index of the layer in the network")
    intermediate_features: IntermediateFeatures
    gradient_flow: GradientFlow
    parameter_statistics: ParameterStatistics

class CrossLayerAnalysis(BaseModel):
    feature_std_gradient: float = Field(..., description="Feature standard deviation gradient")
    gradient_norm_ratio: Dict[str, float] = Field(default_factory=dict, description="Ratio of gradient norms across layers")

    @field_validator('feature_std_gradient')
    @classmethod
    def validate_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Value must be finite (not NaN or Infinity)")
        return v

class MetricsPayload(BaseModel):
    metadata: Metadata
    layer_statistics: List[LayerStatistic] = Field(..., min_length=1, description="List of layer statistics")
    cross_layer_analysis: CrossLayerAnalysis

    @field_validator('layer_statistics')
    @classmethod
    def check_layer_ordering(cls, v):
        """Verify depth_index is sequential starting from 0"""
        if v:
            for i, layer in enumerate(v):
                if layer.depth_index != i:
                    raise ValueError(f"Layer depth_index must be sequential starting from 0. Expected {i}, got {layer.depth_index} for layer '{layer.layer_id}'")
        return v


# ==================== In-Memory Storage ====================

class MetricsStore:
    """Thread-safe in-memory storage for metrics with automatic cleanup"""
    
    def __init__(self, max_runs: int = 10, max_steps_per_run: int = 1000):
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.max_runs = max_runs
        self.max_steps_per_run = max_steps_per_run
        self._lock = asyncio.Lock()
    
    async def add_metrics(self, payload: MetricsPayload):
        async with self._lock:
            run_id = payload.metadata.run_id
            
            if run_id not in self.runs:
                # Cleanup old runs if limit reached
                if len(self.runs) >= self.max_runs:
                    oldest_run = min(self.runs.keys(), 
                                   key=lambda k: self.runs[k]['last_update'])
                    del self.runs[oldest_run]
                
                self.runs[run_id] = {
                    'created_at': datetime.now().isoformat(),
                    'last_update': datetime.now().isoformat(),
                    'steps': []
                }
            
            run = self.runs[run_id]
            
            # Add new step data
            step_data = {
                'step': payload.metadata.global_step,
                'timestamp': payload.metadata.timestamp,
                'batch_size': payload.metadata.batch_size,
                'layers': [layer.model_dump() for layer in payload.layer_statistics],
                'cross_layer': payload.cross_layer_analysis.model_dump()
            }
            
            # Insert in order, avoid duplicates
            existing_idx = next((i for i, s in enumerate(run['steps']) 
                                if s['step'] == step_data['step']), None)
            if existing_idx is not None:
                run['steps'][existing_idx] = step_data
            else:
                run['steps'].append(step_data)
                run['steps'].sort(key=lambda x: x['step'])
            
            # Keep only recent steps
            if len(run['steps']) > self.max_steps_per_run:
                run['steps'] = run['steps'][-self.max_steps_per_run:]
            
            run['last_update'] = datetime.now().isoformat()
            
            return step_data
    
    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self.runs.get(run_id)
    
    async def get_all_runs(self) -> Dict[str, Dict[str, Any]]:
        async with self._lock:
            return {
                run_id: {
                    'created_at': run['created_at'],
                    'last_update': run['last_update'],
                    'step_count': len(run['steps']),
                    'latest_step': run['steps'][-1]['step'] if run['steps'] else None
                }
                for run_id, run in self.runs.items()
            }
    
    async def get_latest_step(self, run_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            run = self.runs.get(run_id)
            if run and run['steps']:
                return run['steps'][-1]
            return None


# Global store instance
store = MetricsStore()


# ==================== WebSocket Manager ====================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        async with self._lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            # Clean up failed connections
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)


manager = ConnectionManager()


# ==================== API Endpoints ====================

@app.post("/api/v1/metrics/layerwise", status_code=202)
async def receive_metrics(payload: MetricsPayload):
    """
    Receive layer-wise metrics from training monitor.
    Returns 202 Accepted immediately, processes asynchronously.
    """
    try:
        # Store metrics
        step_data = await store.add_metrics(payload)

        # Broadcast to connected WebSocket clients
        await manager.broadcast({
            'type': 'new_metrics',
            'run_id': payload.metadata.run_id,
            'data': step_data
        })

        return {"status": "accepted", "run_id": payload.metadata.run_id}

    except ValueError as e:
        # Handle business logic validation errors
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": str(e)}
        )
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "Failed to process metrics"}
        )


@app.get("/api/v1/runs")
async def get_runs():
    """Get list of all active training runs"""
    return await store.get_all_runs()


@app.get("/api/v1/runs/{run_id}")
async def get_run_data(run_id: str):
    """Get full data for a specific run"""
    run = await store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@app.get("/api/v1/runs/{run_id}/latest")
async def get_latest_metrics(run_id: str):
    """Get latest step data for a run"""
    step = await store.get_latest_step(run_id)
    if not step:
        raise HTTPException(status_code=404, detail=f"No data for run '{run_id}'")
    return step


# ==================== WebSocket Endpoint ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time metric updates"""
    await manager.connect(websocket)
    try:
        # Send current runs list on connect
        runs = await store.get_all_runs()
        await websocket.send_json({
            'type': 'initial_runs',
            'data': runs
        })

        # Keep connection alive and handle client messages
        while True:
            try:
                message = await websocket.receive_text()

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    })
                    continue

                # Handle subscription to specific run
                if data.get('action') == 'subscribe_run':
                    run_id = data.get('run_id')
                    run = await store.get_run(run_id)
                    if run:
                        await websocket.send_json({
                            'type': 'run_history',
                            'run_id': run_id,
                            'data': run
                        })
                    else:
                        await websocket.send_json({
                            'type': 'error',
                            'message': f"Run '{run_id}' not found"
                        })

                # Handle ping
                elif data.get('action') == 'ping':
                    await websocket.send_json({'type': 'pong'})

            except json.JSONDecodeError:
                pass  # Already handled above
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


# ==================== Static Files ====================

# Serve the frontend
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def root():
    """Serve the main dashboard"""
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "NN Training Monitor Server", "docs": "/docs"}


# ==================== Health Check ====================

@app.get("/health")
async def health():
    return {"status": "healthy", "active_connections": len(manager.active_connections)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
