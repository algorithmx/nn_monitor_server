"""
Neural Network Training Monitor Server
FastAPI backend with WebSocket support for real-time visualization
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import asyncio
import json
import math
import os


# ==================== Configuration ====================

class ServerConfig(BaseSettings):
    """Server configuration using pydantic-settings for environment variable support"""
    model_config = SettingsConfigDict(
        env_prefix="NN_MONITOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    max_runs: int = Field(default=10, ge=1, le=100, description="Maximum number of concurrent runs to store")
    max_steps_per_run: int = Field(default=1000, ge=10, le=10000, description="Maximum steps to keep per run")
    max_request_size: int = Field(default=2_000_000, ge=100_000, description="Maximum request size in bytes")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    log_level: str = Field(default="info", description="Log level")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")


# Load configuration from environment variables
config = ServerConfig()

app = FastAPI(title="NN Training Monitor", version="1.0.0", max_request_size=config.max_request_size)

# ==================== Data Models ====================

class Metadata(BaseModel):
    run_id: str = Field(..., min_length=1, description="Unique identifier for the training run")
    timestamp: float = Field(..., gt=0, description="Unix epoch timestamp")
    global_step: int = Field(..., ge=0, description="Training step number")
    batch_size: int = Field(..., gt=0, description="Batch size")
    layer_groups: Optional[Dict[str, List[str]]] = Field(None, description="Optional grouping of layers by group name mapping to list of layer_ids")

class IntermediateFeatures(BaseModel):
    activation_std: float = Field(..., ge=0, description="Standard deviation of activations (can be 0 for dead neurons)")
    activation_mean: float = Field(..., description="Mean of activations")
    activation_shape: List[int] = Field(..., min_length=2, description="Shape of activation tensor")
    cross_layer_std_ratio: Optional[float] = Field(None, ge=0, description="Cross-layer standard deviation ratio (can be 0 for vanishing)")

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
    gradient_l2_norm: float = Field(..., ge=0, description="L2 norm of gradients")
    gradient_std: float = Field(..., ge=0, description="Standard deviation of gradients")
    gradient_max_abs: float = Field(..., ge=0, description="Maximum absolute gradient value")

    @field_validator('gradient_l2_norm', 'gradient_std', 'gradient_max_abs')
    @classmethod
    def validate_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Value must be finite (not NaN or Infinity)")
        return v

class WeightStats(BaseModel):
    std: float = Field(..., ge=0, description="Standard deviation of weights (can be 0 for zero initialization)")
    mean: float = Field(..., description="Mean of weights")
    spectral_norm: float = Field(..., ge=0, description="Spectral norm of weight matrix")
    frobenius_norm: float = Field(..., ge=0, description="Frobenius norm of weight matrix")

    @field_validator('std', 'mean', 'spectral_norm', 'frobenius_norm')
    @classmethod
    def validate_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Value must be finite (not NaN or Infinity)")
        return v

class BiasStats(BaseModel):
    std: float = Field(..., ge=0, description="Standard deviation of bias")
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
        """Verify layers are sorted by depth_index and depth_index is non-negative"""
        if v:
            # Check that depth_index values are non-negative
            for i, layer in enumerate(v):
                if layer.depth_index < 0:
                    raise ValueError(f"Layer depth_index must be non-negative, got {layer.depth_index} for layer '{layer.layer_id}'")
            # Check that layers are sorted by depth_index
            for i in range(len(v) - 1):
                if v[i].depth_index > v[i + 1].depth_index:
                    raise ValueError(f"Layers must be sorted by depth_index: {v[i].layer_id} has depth_index {v[i].depth_index} but {v[i + 1].layer_id} has depth_index {v[i + 1].depth_index}")
        return v


# ==================== WebSocket Message Models ====================

class WebSocketMessage(BaseModel):
    """Base WebSocket message model"""
    type: str = Field(..., description="Message type identifier")


class WebSocketSubscribeRun(WebSocketMessage):
    """WebSocket subscription message for a specific run"""
    type: Literal["subscribe_run"] = "subscribe_run"
    run_id: str = Field(..., min_length=1, description="Run ID to subscribe to")


class WebSocketPing(WebSocketMessage):
    """WebSocket ping message"""
    type: Literal["ping"] = "ping"


class WebSocketPong(WebSocketMessage):
    """WebSocket pong response"""
    type: Literal["pong"] = "pong"


class WebSocketNewMetrics(WebSocketMessage):
    """WebSocket message for new metrics"""
    type: Literal["new_metrics"] = "new_metrics"
    run_id: str = Field(..., min_length=1, description="Run ID")
    data: Dict[str, Any] = Field(..., description="Step data")


class WebSocketInitialRuns(WebSocketMessage):
    """WebSocket message with initial runs list"""
    type: Literal["initial_runs"] = "initial_runs"
    data: Dict[str, Any] = Field(..., description="All runs data")


class WebSocketRunHistory(WebSocketMessage):
    """WebSocket message with run history"""
    type: Literal["run_history"] = "run_history"
    run_id: str = Field(..., min_length=1, description="Run ID")
    data: Dict[str, Any] = Field(..., description="Run data")


class WebSocketError(WebSocketMessage):
    """WebSocket error message"""
    type: Literal["error"] = "error"
    message: str = Field(..., min_length=1, description="Error message")


# ==================== Response Models ====================

class MetricsAcceptedResponse(BaseModel):
    """Response for accepted metrics"""
    status: Literal["accepted"] = Field(default="accepted", description="Status indicating metrics were accepted")
    run_id: str = Field(..., min_length=1, description="Run ID")


class RunInfo(BaseModel):
    """Information about a run"""
    created_at: str = Field(..., description="ISO format creation timestamp")
    last_update: str = Field(..., description="ISO format last update timestamp")
    step_count: int = Field(..., ge=0, description="Number of steps recorded")
    latest_step: Optional[int] = Field(None, ge=0, description="Latest step number")


class StepData(BaseModel):
    """Data for a single step"""
    step: int = Field(..., ge=0, description="Step number")
    timestamp: float = Field(..., gt=0, description="Unix epoch timestamp")
    batch_size: int = Field(..., gt=0, description="Batch size")
    layers: List[Dict[str, Any]] = Field(..., min_length=1, description="Layer statistics")
    cross_layer: Dict[str, Any] = Field(..., description="Cross-layer analysis")
    layer_groups: Optional[Dict[str, List[str]]] = Field(None, description="Optional grouping of layers by group name")


class RunData(BaseModel):
    """Full data for a run"""
    created_at: str = Field(..., description="ISO format creation timestamp")
    last_update: str = Field(..., description="ISO format last update timestamp")
    steps: List[StepData] = Field(default_factory=list, description="Step data")


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy"] = Field(default="healthy", description="Health status")
    active_connections: int = Field(..., ge=0, description="Number of active WebSocket connections")


# ==================== Error Models ====================

class ErrorDetail(BaseModel):
    """Error detail model"""
    error: Literal["validation_error", "internal_error", "not_found"] = Field(..., description="Error type")
    message: str = Field(..., min_length=1, description="Human-readable error message")


# ==================== In-Memory Storage ====================

class MetricsStore:
    """Thread-safe in-memory storage for metrics with automatic cleanup"""

    def __init__(self, max_runs: int = 10, max_steps_per_run: int = 1000):
        self.runs: Dict[str, RunData] = {}
        self.max_runs = max_runs
        self.max_steps_per_run = max_steps_per_run
        self._lock = asyncio.Lock()

    async def add_metrics(self, payload: MetricsPayload) -> StepData:
        async with self._lock:
            run_id = payload.metadata.run_id

            if run_id not in self.runs:
                # Cleanup old runs if limit reached
                if len(self.runs) >= self.max_runs:
                    oldest_run = min(self.runs.keys(),
                                   key=lambda k: self.runs[k].last_update)
                    del self.runs[oldest_run]

                self.runs[run_id] = RunData(
                    created_at=datetime.now().isoformat(),
                    last_update=datetime.now().isoformat(),
                    steps=[]
                )

            run = self.runs[run_id]

            # Add new step data using pydantic model
            step_data = StepData(
                step=payload.metadata.global_step,
                timestamp=payload.metadata.timestamp,
                batch_size=payload.metadata.batch_size,
                layers=[layer.model_dump() for layer in payload.layer_statistics],
                cross_layer=payload.cross_layer_analysis.model_dump(),
                layer_groups=payload.metadata.layer_groups
            )

            # Insert in order, avoid duplicates
            existing_idx = next((i for i, s in enumerate(run.steps)
                                if s.step == step_data.step), None)
            if existing_idx is not None:
                run.steps[existing_idx] = step_data
            else:
                run.steps.append(step_data)
                run.steps.sort(key=lambda x: x.step)

            # Keep only recent steps
            if len(run.steps) > self.max_steps_per_run:
                run.steps = run.steps[-self.max_steps_per_run:]

            run.last_update = datetime.now().isoformat()

            return step_data

    async def get_run(self, run_id: str) -> Optional[RunData]:
        async with self._lock:
            return self.runs.get(run_id)

    async def get_all_runs(self) -> Dict[str, RunInfo]:
        async with self._lock:
            return {
                run_id: RunInfo(
                    created_at=run.created_at,
                    last_update=run.last_update,
                    step_count=len(run.steps),
                    latest_step=run.steps[-1].step if run.steps else None
                )
                for run_id, run in self.runs.items()
            }

    async def get_latest_step(self, run_id: str) -> Optional[StepData]:
        async with self._lock:
            run = self.runs.get(run_id)
            if run and run.steps:
                return run.steps[-1]
            return None


# Global store instance
store = MetricsStore(max_runs=config.max_runs, max_steps_per_run=config.max_steps_per_run)


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

@app.post("/api/v1/metrics/layerwise", status_code=202, response_model=MetricsAcceptedResponse)
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
            'data': step_data.model_dump()
        })

        return MetricsAcceptedResponse(run_id=payload.metadata.run_id)

    except ValueError as e:
        # Handle business logic validation errors
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(error="validation_error", message=str(e)).model_dump()
        )
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(error="internal_error", message="Failed to process metrics").model_dump()
        )


@app.get("/api/v1/runs", response_model=Dict[str, RunInfo])
async def get_runs():
    """Get list of all active training runs"""
    return await store.get_all_runs()


@app.get("/api/v1/runs/{run_id}", response_model=RunData)
async def get_run_data(run_id: str):
    """Get full data for a specific run"""
    run = await store.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(error="not_found", message=f"Run '{run_id}' not found").model_dump()
        )
    return run


@app.get("/api/v1/runs/{run_id}/latest", response_model=StepData)
async def get_latest_metrics(run_id: str):
    """Get latest step data for a run"""
    step = await store.get_latest_step(run_id)
    if not step:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(error="not_found", message=f"No data for run '{run_id}'").model_dump()
        )
    return step


# ==================== WebSocket Endpoint ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time metric updates"""
    await manager.connect(websocket)
    try:
        # Send current runs list on connect
        runs = await store.get_all_runs()
        # Convert Pydantic models to dicts for JSON serialization
        runs_dict = {run_id: run.model_dump() for run_id, run in runs.items()}
        await websocket.send_json({
            'type': 'initial_runs',
            'data': runs_dict
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
                            'data': run.model_dump()
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

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(active_connections=len(manager.active_connections))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)
