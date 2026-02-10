"""
Neural Network Training Monitor Server
FastAPI backend with WebSocket support for real-time visualization
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import json
import os

app = FastAPI(title="NN Training Monitor", version="1.0.0")

# ==================== Data Models ====================

class Metadata(BaseModel):
    run_id: str
    timestamp: float
    global_step: int
    batch_size: int

class IntermediateFeatures(BaseModel):
    activation_std: float
    activation_mean: float
    activation_shape: List[int]
    cross_layer_std_ratio: Optional[float] = None

class GradientFlow(BaseModel):
    gradient_l2_norm: float
    gradient_std: float
    gradient_max_abs: float

class WeightStats(BaseModel):
    std: float
    mean: float
    spectral_norm: float
    frobenius_norm: float

class BiasStats(BaseModel):
    std: float
    mean_abs: float

class ParameterStatistics(BaseModel):
    weight: WeightStats
    bias: BiasStats

class LayerStatistic(BaseModel):
    layer_id: str
    layer_type: str
    depth_index: int
    intermediate_features: IntermediateFeatures
    gradient_flow: GradientFlow
    parameter_statistics: ParameterStatistics

class CrossLayerAnalysis(BaseModel):
    feature_std_gradient: float
    gradient_norm_ratio: Dict[str, float]

class MetricsPayload(BaseModel):
    metadata: Metadata
    layer_statistics: List[LayerStatistic]
    cross_layer_analysis: CrossLayerAnalysis


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
                'layers': [layer.dict() for layer in payload.layer_statistics],
                'cross_layer': payload.cross_layer_analysis.dict()
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                data = json.loads(message)
                
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
                
                # Handle ping
                elif data.get('action') == 'ping':
                    await websocket.send_json({'type': 'pong'})
                    
            except json.JSONDecodeError:
                pass
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
