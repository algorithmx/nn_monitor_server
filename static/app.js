const state = {
    ws: null,
    runs: {},
    currentRun: null,
    runData: null,
    history: {}, // { layerId: [{step, actStd, gradNorm}, ...] }
    // History navigation state
    isLiveMode: true,           // true = showing live updates, false = paused on specific step
    selectedStep: null,         // step number being viewed (null = latest when liveMode)
    stepsList: [],              // sorted list of all available step numbers for current run
    lastSeenStep: null,         // track the last step we saw for smart resume
    lastViewedStep: null,       // track the last step the user was viewing before live mode
    isTrainingActive: true,     // track if training is still receiving updates
    // Visualization config
    vizConfig: {
        gradScale: 1.0          // vertical scale multiplier for gradient norm in plots
    }
};

const CONFIG = {
    historyLen: 50,
    trainingTimeoutMs: 10000,
    maxRunDataSteps: 5000,     // Max steps to keep in runData.steps
    maxLayerHistory: 10000,    // Max history entries per layer
    thresholds: {
        gradNorm: { low: 0.001, high: 10 },
        gradMax: { high: 10 },
        actRatio: { low: 0.5, high: 2 }
    }
};

// WebSocket
function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(`${proto}//${location.host}/ws`);

    state.ws.onopen = () => setStatus(true, 'live');
    state.ws.onclose = () => {
        setStatus(false, 'reconnecting');
        // Try to reconnect after 2 seconds
        setTimeout(() => {
            if (state.ws?.readyState === WebSocket.CLOSED) {
                connect();
            }
        }, 2000);
    };
    state.ws.onerror = () => setStatus(false, 'error');
    state.ws.onmessage = (e) => {
        let data;
        try {
            data = JSON.parse(e.data);
        } catch (err) {
            console.error('Failed to parse WebSocket message:', err, e.data);
            return;
        }
        try {
            handleMessage(data);
        } catch (err) {
            console.error('Failed to handle WebSocket message:', err, data);
        }
    };
}

function setStatus(connected, text) {
    document.getElementById('statusDot').classList.toggle('disconnected', !connected);
    document.getElementById('statusText').textContent = text;
}

function handleMessage(msg) {
    switch(msg.type) {
        case 'initial_runs':
            state.runs = msg.data;
            updateSelect();
            // Update status to live since we successfully received data
            setStatus(true, 'live');
            break;

        case 'new_metrics':
            if (!state.runs[msg.run_id]) {
                state.runs[msg.run_id] = {
                    created_at: new Date().toISOString(),
                    last_update: new Date().toISOString(),
                    step_count: 0
                };
                updateSelect();
            }

            if (!state.currentRun && Object.keys(state.runs).length) {
                selectRun(msg.run_id);
            }

            if (msg.run_id === state.currentRun) {
                resetTrainingTimeout();
                addStep(msg.data);

                // Update step_count based on actual steps array
                if (state.runData && state.runData.steps) {
                    state.runs[msg.run_id].step_count = state.runData.steps.length;
                    state.runs[msg.run_id].last_update = new Date().toISOString();
                    updateSelect();
                }

                // If in paused mode, update UI elements but don't disrupt the user's view
                if (!state.isLiveMode) {
                    updateStepsList();
                    updateHistoryUI();
                }
            }
            break;

        case 'run_history':
            // Apply sliding window to loaded history
            if (msg.data.steps.length > CONFIG.maxRunDataSteps) {
                msg.data.steps = msg.data.steps.slice(-CONFIG.maxRunDataSteps);
            }
            state.runData = msg.data;

            // Rebuild per-layer history from loaded steps
            state.history = {};
            msg.data.steps.forEach(step => {
                step.layers.forEach(layer => {
                    if (!layer.layer_id) return;
                    if (!state.history[layer.layer_id]) {
                        state.history[layer.layer_id] = [];
                    }
                    state.history[layer.layer_id].push({
                        step: step.step,
                        actStd: layer.intermediate_features?.activation_std ?? 0,
                        gradNorm: layer.gradient_flow?.gradient_l2_norm ?? 0
                    });
                });
            });

            if (msg.data.steps.length > 0) {
                state.lastSeenStep = msg.data.steps[msg.data.steps.length - 1].step;
            }
            // Update step_count in runs info
            if (state.runs[msg.run_id] && state.runData) {
                state.runs[msg.run_id].step_count = state.runData.steps.length;
                updateSelect();
            }
            break;
    }
}

function updateSelect() {
    const sel = document.getElementById('runSelect');
    const cur = sel.value;
    sel.innerHTML = '<option value="">select run...</option>';
    Object.entries(state.runs).forEach(([id, info]) => {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `${id} (${info.step_count || 0} steps)`;
        sel.appendChild(opt);
    });
    if (cur && state.runs[cur]) sel.value = cur;
}

function selectRun(id) {
    state.currentRun = id;
    document.getElementById('runSelect').value = id;
    state.history = {};
    state.runData = null;  // Clear previous run data to free memory

    // Reset history navigation state
    state.isLiveMode = true;
    state.selectedStep = null;
    state.stepsList = [];
    state.isTrainingActive = true;
    state.lastSeenStep = null;
    state.lastViewedStep = null;
    resetTrainingTimeout();

    updateHistoryUI();

    // Safely check WebSocket state before sending
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        try {
            state.ws.send(JSON.stringify({ action: 'subscribe_run', run_id: id }));
        } catch (err) {
            console.error('Failed to send subscribe_run message:', err);
        }
    }
}

document.getElementById('runSelect').addEventListener('change', (e) => {
    if (e.target.value) selectRun(e.target.value);
});

// Toggle between live and paused mode
function toggleLiveMode() {
    const wasPaused = !state.isLiveMode;

    if (wasPaused) {
        // Returning to live mode - check if training is still active
        if (!state.isTrainingActive && state.lastViewedStep !== null) {
            // Training ended, stay on last viewed step (remain in paused mode)
            state.selectedStep = state.lastViewedStep;
            state.isLiveMode = false;
        } else {
            // Training is active, go to live mode
            state.isLiveMode = true;
            state.selectedStep = null;
            state.lastViewedStep = null;
        }
    } else {
        // Pausing - capture current step
        const steps = state.runData?.steps || [];
        if (steps.length > 0) {
            state.selectedStep = steps[steps.length - 1].step;
        }
        state.isLiveMode = false;
        state.lastViewedStep = null;
    }

    updateHistoryUI();
    render();
}

// Update UI based on current mode
function updateHistoryUI() {
    const modeBadge = document.getElementById('modeBadge');
    const modeText = document.getElementById('modeText');
    const btnLivePause = document.getElementById('btnLivePause');
    const navControls = document.getElementById('navControls');
    const stepSlider = document.getElementById('stepSlider');
    const stepDisplay = document.getElementById('stepDisplay');

    const steps = state.runData?.steps || [];
    const currentStep = state.isLiveMode ? (steps[steps.length - 1]?.step ?? null) : state.selectedStep;
    const totalSteps = steps.length;
    const stepIndex = state.stepsList.indexOf(currentStep);

    if (state.isLiveMode) {
        // Live mode
        modeBadge.className = 'mode-badge live';
        const trainingStatus = state.isTrainingActive ? 'Live' : 'Live (ended)';
        modeText.textContent = trainingStatus;
        btnLivePause.className = 'btn-live live-active';
        btnLivePause.textContent = '⏸ Pause';

        navControls.style.opacity = '0.3';
        navControls.style.pointerEvents = 'none';
        stepSlider.disabled = true;

        stepDisplay.textContent = steps.length > 0 ? `step ${steps[steps.length - 1].step}` : '—';
    } else {
        // Paused mode
        const stepLabel = totalSteps > 0 ? `${stepIndex + 1} / ${totalSteps}` : '—';
        modeBadge.className = 'mode-badge paused';
        modeText.textContent = `Paused @ ${stepLabel}`;
        btnLivePause.className = 'btn-live paused-active';
        btnLivePause.textContent = '▶ Live';

        navControls.style.opacity = '1';
        navControls.style.pointerEvents = 'auto';
        stepSlider.disabled = false;

        stepDisplay.textContent = currentStep !== null ? `step ${currentStep}` : '—';

        // Update slider
        if (state.stepsList.length > 0) {
            stepSlider.min = 0;
            stepSlider.max = state.stepsList.length - 1;
            stepSlider.value = stepIndex;
        }

        // Update nav button states
        document.getElementById('btnFirst').disabled = stepIndex <= 0;
        document.getElementById('btnPrev').disabled = stepIndex <= 0;
        document.getElementById('btnNext').disabled = stepIndex >= totalSteps - 1;
        document.getElementById('btnLast').disabled = stepIndex >= totalSteps - 1;
    }
}

// Navigate to a specific step by direction
function navigateToStep(direction) {
    const steps = state.runData?.steps || [];
    if (steps.length === 0) return;

    const currentIndex = state.stepsList.indexOf(state.selectedStep);
    let newIndex = currentIndex;

    switch(direction) {
        case 'first':
            newIndex = 0;
            break;
        case 'prev':
            newIndex = Math.max(0, currentIndex - 1);
            break;
        case 'next':
            newIndex = Math.min(state.stepsList.length - 1, currentIndex + 1);
            break;
        case 'last':
            newIndex = state.stepsList.length - 1;
            break;
    }

    navigateToStepIndex(newIndex);
}

// Navigate to a specific step by index
function navigateToStepIndex(index) {
    if (index < 0 || index >= state.stepsList.length) return;

    state.selectedStep = state.stepsList[index];
    state.lastViewedStep = state.selectedStep; // Save for smart resume
    state.isLiveMode = false;
    updateHistoryUI();
    render();
}

// Update steps list when new data arrives
function updateStepsList() {
    const steps = state.runData?.steps || [];
    state.stepsList = steps.map(s => s.step).sort((a, b) => a - b);

    // Update slider range
    const slider = document.getElementById('stepSlider');
    if (state.stepsList.length > 0) {
        slider.min = 0;
        slider.max = state.stepsList.length - 1;
    }
}

// Detect when training has ended (no new data for 10 seconds)
let trainingTimeout = null;
function resetTrainingTimeout() {
    if (trainingTimeout) clearTimeout(trainingTimeout);
    trainingTimeout = setTimeout(() => {
        state.isTrainingActive = false;
        if (state.isLiveMode) {
            updateHistoryUI();
        }
    }, CONFIG.trainingTimeoutMs);
}

// Live/Pause toggle
document.getElementById('btnLivePause').addEventListener('click', toggleLiveMode);

// Navigation buttons
document.getElementById('btnFirst').addEventListener('click', () => navigateToStep('first'));
document.getElementById('btnPrev').addEventListener('click', () => navigateToStep('prev'));
document.getElementById('btnNext').addEventListener('click', () => navigateToStep('next'));
document.getElementById('btnLast').addEventListener('click', () => navigateToStep('last'));

// Slider
document.getElementById('stepSlider').addEventListener('input', (e) => {
    const index = parseInt(e.target.value);
    if (index >= 0 && index < state.stepsList.length) {
        navigateToStepIndex(index);
    }
});

function addStep(stepData) {
    // Validate required fields to prevent runtime errors
    if (!stepData || typeof stepData !== 'object') {
        console.error('Invalid step data: expected object, got', typeof stepData);
        return;
    }
    if (!Array.isArray(stepData.layers)) {
        console.error('Invalid step data: layers must be an array');
        return;
    }
    if (typeof stepData.step !== 'number' || stepData.step < 0) {
        console.error('Invalid step data: step must be a non-negative number');
        return;
    }
    
    if (!state.runData) {
        state.runData = { steps: [] };
    }

    const idx = state.runData.steps.findIndex(s => s.step === stepData.step);
    if (idx >= 0) {
        state.runData.steps[idx] = stepData;
    } else {
        state.runData.steps.push(stepData);
        state.runData.steps.sort((a, b) => a.step - b.step);
        // Sliding window: keep only the most recent steps
        if (state.runData.steps.length > CONFIG.maxRunDataSteps) {
            state.runData.steps = state.runData.steps.slice(-CONFIG.maxRunDataSteps);
        }
    }

    // Update per-layer history for pulse viz - limit to prevent memory exhaustion
    stepData.layers.forEach(layer => {
        if (!layer.layer_id) return;  // Skip layers without ID
        if (!state.history[layer.layer_id]) {
            state.history[layer.layer_id] = [];
        }
        state.history[layer.layer_id].push({
            step: stepData.step,
            actStd: layer.intermediate_features?.activation_std ?? 0,
            gradNorm: layer.gradient_flow?.gradient_l2_norm ?? 0
        });
        // Limit history - use splice for in-place removal with hysteresis
        const history = state.history[layer.layer_id];
        if (history.length > CONFIG.maxLayerHistory + 500) {
            history.splice(0, history.length - CONFIG.maxLayerHistory);
        }
    });

    // Track training activity
    state.lastSeenStep = stepData.step;

    updateStepsList();

    render();
}

// Sanitize layer ID for use as HTML element ID
// Uses a mapping approach to avoid collisions (e.g., 'a.b' and 'a/b' must be different)
function sanitizeLayerId(layerId) {
    return layerId
        .replace(/_/g, '_u_')      // Escape underscores first
        .replace(/\./g, '_d_')     // Dots become _d_
        .replace(/\//g, '_s_')     // Slashes become _s_
        .replace(/[^a-zA-Z0-9]/g, '_'); // Other non-alphanumeric become _
}

function renderLayer(layer) {
    const health = assessHealth(layer);
    return `
        <div class="layer ${health.class}">
            <div class="layer-header">
                <div>
                    <div class="layer-name" data-tooltip="Layer identifier: ${layer.layer_id}">${layer.layer_id}</div>
                    <div class="layer-type" data-tooltip="Layer type: ${layer.layer_type}">${layer.layer_type}</div>
                </div>
            </div>
            <canvas class="pulse-viz" id="pulse-${sanitizeLayerId(layer.layer_id)}" data-tooltip="Historical trend: Green line = activation std, Purple line = gradient norm"></canvas>
            <div class="metrics">
                <div class="metric" data-tooltip="Standard deviation of layer activations. Measures the spread/variability of activations.">
                    <div class="metric-value ${health.act}">${layer.intermediate_features.activation_std.toFixed(4)}</div>
                    <div class="metric-label">act std</div>
                </div>
                <div class="metric" data-tooltip="L2 norm of gradients flowing through the layer. Indicates how much weights are being updated. Low values may indicate vanishing gradients.">
                    <div class="metric-value ${health.grad}">${layer.gradient_flow.gradient_l2_norm.toFixed(4)}</div>
                    <div class="metric-label">grad norm</div>
                </div>
                <div class="metric" data-tooltip="Ratio of this layer's activation std to the previous layer's. Shows how signal propagation changes across layers.">
                    <div class="metric-value ${health.ratio}">${layer.intermediate_features.cross_layer_std_ratio?.toFixed(3) ?? '—'}</div>
                    <div class="metric-label">ratio</div>
                </div>
            </div>
        </div>
    `;
}

function render() {
    if (!state.runData?.steps.length) return;

    const steps = state.runData.steps;

    // Determine which step to display
    let displayStep;
    if (state.isLiveMode) {
        displayStep = steps[steps.length - 1];
    } else {
        const stepData = steps.find(s => s.step === state.selectedStep);
        displayStep = stepData || steps[steps.length - 1];
    }

    if (!displayStep) return;

    // Update summary
    document.getElementById('stepVal').textContent = displayStep.step;
    document.getElementById('layerVal').textContent = displayStep.layers.length;
    document.getElementById('timeVal').textContent = new Date(displayStep.timestamp * 1000).toLocaleTimeString();

    // Update mode display if in live mode (show latest step info)
    if (state.isLiveMode) {
        document.getElementById('stepDisplay').textContent = `step ${displayStep.step}`;
    }

    // Check alerts (only in live mode)
    if (state.isLiveMode) {
        checkAlerts(displayStep);
    } else {
        // Hide alerts when viewing history
        document.getElementById('alerts').classList.remove('visible');
    }

    // Render layers
    const container = document.getElementById('layers');
    const layerGroups = displayStep.layer_groups || null;

    if (layerGroups && Object.keys(layerGroups).length > 0) {
        // Render with grouping
        const groupedLayers = {};
        const ungroupedLayers = [];

        // Initialize all groups
        Object.keys(layerGroups).forEach(groupName => {
            groupedLayers[groupName] = [];
        });

        // Distribute layers into groups
        displayStep.layers.forEach(layer => {
            let assigned = false;
            for (const [groupName, layerIds] of Object.entries(layerGroups)) {
                if (layerIds.includes(layer.layer_id)) {
                    groupedLayers[groupName].push(layer);
                    assigned = true;
                    break;
                }
            }
            if (!assigned) {
                ungroupedLayers.push(layer);
            }
        });

        // Build HTML with groups
        let html = '';
        for (const [groupName, layers] of Object.entries(groupedLayers)) {
            if (layers.length > 0) {
                html += `
                    <div class="group-section">
                        <div class="group-header">${groupName}</div>
                        <div class="layer-grid">
                            ${layers.map(layer => renderLayer(layer)).join('')}
                        </div>
                    </div>
                `;
            }
        }

        // Add ungrouped layers if any
        if (ungroupedLayers.length > 0) {
            html += `
                <div class="group-section">
                    <div class="group-header">Ungrouped</div>
                    <div class="layer-grid">
                        ${ungroupedLayers.map(layer => renderLayer(layer)).join('')}
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
    } else {
        // Render without grouping (flat grid)
        container.className = 'layers';
        container.innerHTML = `<div class="layer-grid">${displayStep.layers.map(layer => renderLayer(layer)).join('')}</div>`;
    }

    // Draw pulse lines with historical context
    displayStep.layers.forEach(layer => {
        drawPulse(layer.layer_id, displayStep.step);
        // Add click handler for modal
        const canvas = document.getElementById(`pulse-${sanitizeLayerId(layer.layer_id)}`);
        if (canvas) {
            canvas.onclick = () => openModal(layer.layer_id, layer.layer_type, displayStep.step);
        }
    });
}

function assessHealth(layer) {
    const gradNorm = layer.gradient_flow.gradient_l2_norm;
    const gradMax = layer.gradient_flow.gradient_max_abs;
    const ratio = layer.intermediate_features.cross_layer_std_ratio;

    let health = 'good';
    let act = 'good';
    let grad = 'good';
    let ratioClass = 'good';

    if (gradNorm < CONFIG.thresholds.gradNorm.low) {
        health = 'critical';
        grad = 'bad';
    }
    if (gradMax > CONFIG.thresholds.gradMax.high) {
        health = 'critical';
        grad = 'bad';
    }
    if (ratio !== null) {
        if (ratio < CONFIG.thresholds.actRatio.low) {
            health = health === 'good' ? 'warning' : health;
            ratioClass = 'bad';
        }
        if (ratio > CONFIG.thresholds.actRatio.high) {
            health = health === 'good' ? 'warning' : health;
            ratioClass = 'warn';
        }
    }

    const classMap = { good: '', warning: 'warning', critical: 'critical' };
    return { class: classMap[health], act, grad, ratio: ratioClass };
}

function drawPulse(layerId, currentStep) {
    const canvas = document.getElementById(`pulse-${sanitizeLayerId(layerId)}`);
    if (!canvas) return;

    // Get all history from beginning - no 50 step limit
    const allHistory = state.history[layerId] || [];

    // Show all data from the beginning up to and including currentStep
    let pulseHistory = [];
    if (allHistory.length > 0) {
        const currentStepIndex = allHistory.findIndex(h => h.step === currentStep);
        if (currentStepIndex >= 0) {
            // Show ALL steps from beginning up to current step
            pulseHistory = allHistory.slice(0, currentStepIndex + 1);
        } else {
            // Current step not in history yet, use all available history
            pulseHistory = allHistory;
        }
    }

    if (pulseHistory.length < 2) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);

    // Find min/max for scaling - activation and gradient scale independently
    let maxAct = 0, maxGrad = 0;
    pulseHistory.forEach(pt => {
        maxAct = Math.max(maxAct, pt.actStd);
        maxGrad = Math.max(maxGrad, pt.gradNorm);
    });
    maxAct = Math.max(maxAct, 0.001);
    // Apply gradient scale to determine effective max for gradient plotting
    const effectiveGradMax = Math.max(maxGrad * state.vizConfig.gradScale, 0.001);

    // Draw activation line
    ctx.strokeStyle = '#4a9';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    pulseHistory.forEach((pt, i) => {
        const x = (i / (pulseHistory.length - 1)) * w;
        const y = h - (pt.actStd / maxAct) * h * 0.8 - h * 0.1;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw gradient line - scaled by configured factor
    ctx.strokeStyle = '#a371f7';
    ctx.lineWidth = 1;
    ctx.beginPath();

    pulseHistory.forEach((pt, i) => {
        const x = (i / (pulseHistory.length - 1)) * w;
        // Apply gradient scale multiplier for visualization
        const scaledGradNorm = pt.gradNorm * state.vizConfig.gradScale;
        const y = h - (scaledGradNorm / effectiveGradMax) * h * 0.8 - h * 0.1;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw outer frame
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    // Draw left Y-axis ticks (green, for activation)
    ctx.strokeStyle = '#4a9';
    ctx.fillStyle = '#4a9';
    ctx.font = '10px SF Mono, Menlo, Monaco, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const tickLength = 5;
    const leftPadding = 4;
    for (let i = 0; i <= 2; i++) {
        const frac = i / 2;
        const y = h * 0.1 + frac * h * 0.8;
        // Tick mark
        ctx.beginPath();
        ctx.moveTo(0.5, y);
        ctx.lineTo(0.5 + tickLength, y);
        ctx.stroke();
        // Label
        const value = maxAct * (1 - frac);
        const label = formatValue(value);
        ctx.fillText(label, leftPadding, y);
    }

    // Draw right Y-axis ticks (purple, for gradient)
    ctx.strokeStyle = '#a371f7';
    ctx.fillStyle = '#a371f7';
    ctx.textAlign = 'right';
    const rightPadding = w - 4;
    for (let i = 0; i <= 2; i++) {
        const frac = i / 2;
        const y = h * 0.1 + frac * h * 0.8;
        // Tick mark
        ctx.beginPath();
        ctx.moveTo(w - 0.5 - tickLength, y);
        ctx.lineTo(w - 0.5, y);
        ctx.stroke();
        // Label
        const value = effectiveGradMax * (1 - frac);
        const label = formatValue(value);
        ctx.fillText(label, rightPadding, y);
    }
}

// Helper function to format values concisely
function formatValue(val) {
    if (val >= 1000) return (val / 1000).toFixed(1) + 'k';
    if (val >= 100) return val.toFixed(0);
    if (val >= 10) return val.toFixed(1);
    if (val >= 1) return val.toFixed(2);
    if (val >= 0.01) return val.toFixed(3);
    return val.toExponential(1);
}

function checkAlerts(step) {
    const alerts = [];

    step.layers.forEach(layer => {
        const gradNorm = layer.gradient_flow.gradient_l2_norm;
        const gradMax = layer.gradient_flow.gradient_max_abs;
        const ratio = layer.intermediate_features.cross_layer_std_ratio;

        if (gradNorm < 0.001) {
            alerts.push({ type: 'critical', msg: `vanishing: ${layer.layer_id}` });
        }
        if (gradMax > 10) {
            alerts.push({ type: 'critical', msg: `exploding: ${layer.layer_id}` });
        }
        if (ratio !== null && ratio < 0.1) {
            alerts.push({ type: 'warning', msg: `severe drop: ${layer.layer_id}` });
        }
    });

    const alertDiv = document.getElementById('alerts');
    if (alerts.length) {
        alertDiv.innerHTML = alerts.map(a => `
            <div class="alert ${a.type}">
                <span class="alert-icon">${a.type === 'critical' ? '⚠' : '◷'}</span>
                <span>${a.msg}</span>
            </div>
        `).join('');
        alertDiv.classList.add('visible');
    } else {
        alertDiv.classList.remove('visible');
    }
}

// Modal functionality for zoomed pulse visualization
const modal = document.getElementById('pulseModal');
const modalTitle = document.getElementById('modalTitle');
const modalCanvas = document.getElementById('modalCanvas');
const modalClose = document.getElementById('modalClose');
let currentModalLayerId = null;
let currentModalStep = null;

function openModal(layerId, layerType, currentStep) {
    currentModalLayerId = layerId;
    currentModalStep = currentStep;
    modalTitle.textContent = `${layerId} (${layerType}) - Pulse Visualization`;
    modal.classList.add('visible');
    drawModalPulse();
}

function closeModal() {
    modal.classList.remove('visible');
    currentModalLayerId = null;
    currentModalStep = null;
}

function drawModalPulse() {
    if (!currentModalLayerId || !currentModalStep) return;

    const allHistory = state.history[currentModalLayerId] || [];

    // Show ALL data from beginning up to and including currentStep
    let pulseHistory = [];
    if (allHistory.length > 0) {
        const currentStepIndex = allHistory.findIndex(h => h.step === currentModalStep);
        if (currentStepIndex >= 0) {
            // Show ALL steps from beginning up to current step
            pulseHistory = allHistory.slice(0, currentStepIndex + 1);
        } else {
            // Current step not in history yet, use all available history
            pulseHistory = allHistory;
        }
    }

    if (pulseHistory.length < 2) return;

    const container = modalCanvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();

    modalCanvas.width = rect.width * dpr;
    modalCanvas.height = rect.height * dpr;
    modalCanvas.style.width = rect.width + 'px';
    modalCanvas.style.height = rect.height + 'px';

    const ctx = modalCanvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Define padding for axes
    const leftPadding = 50;
    const rightPadding = 50;
    const bottomPadding = 25;
    const topPadding = 10;

    const plotW = w - leftPadding - rightPadding;
    const plotH = h - topPadding - bottomPadding;
    const plotX = leftPadding;
    const plotY = topPadding;

    ctx.clearRect(0, 0, w, h);

    // Find min/max for scaling - activation and gradient scale independently
    let maxAct = 0, maxGrad = 0;
    pulseHistory.forEach(pt => {
        maxAct = Math.max(maxAct, pt.actStd);
        maxGrad = Math.max(maxGrad, pt.gradNorm);
    });
    maxAct = Math.max(maxAct, 0.001);
    // Apply gradient scale to determine effective max for gradient plotting
    const effectiveGradMax = Math.max(maxGrad * state.vizConfig.gradScale, 0.001);

    // Draw horizontal grid lines (aligned with ticks)
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = plotY + (i / 4) * plotH;
        ctx.beginPath();
        ctx.moveTo(plotX, y);
        ctx.lineTo(plotX + plotW, y);
        ctx.stroke();
    }

    // Draw activation line
    ctx.strokeStyle = '#4a9';
    ctx.lineWidth = 2.5;
    ctx.beginPath();

    pulseHistory.forEach((pt, i) => {
        const x = plotX + (i / (pulseHistory.length - 1)) * plotW;
        const y = plotY + plotH - (pt.actStd / maxAct) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw gradient line - scaled by configured factor
    ctx.strokeStyle = '#a371f7';
    ctx.lineWidth = 2;
    ctx.beginPath();

    pulseHistory.forEach((pt, i) => {
        const x = plotX + (i / (pulseHistory.length - 1)) * plotW;
        // Apply gradient scale multiplier for visualization
        const scaledGradNorm = pt.gradNorm * state.vizConfig.gradScale;
        const y = plotY + plotH - (scaledGradNorm / effectiveGradMax) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw left Y-axis (green, for activation)
    ctx.strokeStyle = '#4a9';
    ctx.fillStyle = '#4a9';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(plotX, plotY);
    ctx.lineTo(plotX, plotY + plotH);
    ctx.stroke();

    // Left Y-axis ticks and labels
    ctx.font = '12px SF Mono, Menlo, Monaco, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
        const frac = i / 4;
        const y = plotY + plotH - frac * plotH;
        // Tick mark
        ctx.beginPath();
        ctx.moveTo(plotX - 5, y);
        ctx.lineTo(plotX, y);
        ctx.stroke();
        // Label
        const value = maxAct * frac;
        const label = formatValue(value);
        ctx.fillText(label, plotX - 8, y);
    }

    // Draw right Y-axis (purple, for gradient)
    ctx.strokeStyle = '#a371f7';
    ctx.fillStyle = '#a371f7';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(plotX + plotW, plotY);
    ctx.lineTo(plotX + plotW, plotY + plotH);
    ctx.stroke();

    // Right Y-axis ticks and labels
    ctx.textAlign = 'left';
    for (let i = 0; i <= 4; i++) {
        const frac = i / 4;
        const y = plotY + plotH - frac * plotH;
        // Tick mark
        ctx.beginPath();
        ctx.moveTo(plotX + plotW, y);
        ctx.lineTo(plotX + plotW + 5, y);
        ctx.stroke();
        // Label
        const value = effectiveGradMax * frac;
        const label = formatValue(value);
        ctx.fillText(label, plotX + plotW + 8, y);
    }

    // Draw bottom X-axis
    ctx.strokeStyle = '#888';
    ctx.fillStyle = '#888';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(plotX, plotY + plotH);
    ctx.lineTo(plotX + plotW, plotY + plotH);
    ctx.stroke();

    // X-axis ticks and step labels
    ctx.font = '12px SF Mono, Menlo, Monaco, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // Show first, middle, last step numbers
    if (pulseHistory.length > 2) {
        const stepPositions = [
            { idx: 0, x: plotX },
            { idx: Math.floor(pulseHistory.length / 2), x: plotX + plotW / 2 },
            { idx: pulseHistory.length - 1, x: plotX + plotW }
        ];
        stepPositions.forEach(pos => {
            const y = plotY + plotH;
            // Tick mark
            ctx.beginPath();
            ctx.moveTo(pos.x, y);
            ctx.lineTo(pos.x, y + 5);
            ctx.stroke();
            // Label
            ctx.fillText(pulseHistory[pos.idx].step.toString(), pos.x, y + 8);
        });
    }
}

// Modal event listeners
modalClose.addEventListener('click', closeModal);
modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
});

// Note: Config panel variables defined earlier with modal handlers

// Unified Escape key handler for all modals/panels
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    // Close topmost layer first (config panel has higher z-index)
    if (configPanel && configPanel.classList.contains('visible')) {
        closeConfig();
        return;
    }
    if (modal && modal.classList.contains('visible')) {
        closeModal();
    }
});

// Configuration panel
const configPanel = document.getElementById('configPanel');
const configClose = document.getElementById('configClose');
const btnSettings = document.getElementById('btnSettings');
const gradScaleSlider = document.getElementById('gradScaleSlider');
const gradScaleValue = document.getElementById('gradScaleValue');

function openConfig() {
    configPanel.classList.add('visible');
}

function closeConfig() {
    configPanel.classList.remove('visible');
    // Re-render plots after config closes to ensure changes take effect
    if (state.runData) render();
    // Also update modal if it's open
    if (modal.classList.contains('visible')) {
        drawModalPulse();
    }
}

btnSettings.addEventListener('click', openConfig);
configClose.addEventListener('click', closeConfig);
configPanel.addEventListener('click', (e) => {
    if (e.target === configPanel) closeConfig();
});

// Gradient scale slider
gradScaleSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    state.vizConfig.gradScale = value;
    gradScaleValue.textContent = value.toFixed(1) + 'x';
    // Re-render plots with new scale
    if (state.runData) render();
    // Also update modal if it's open
    if (modal.classList.contains('visible')) {
        drawModalPulse();
    }
});



connect();

// Debounced resize handler to avoid excessive re-renders
let resizeTimeout = null;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        if (state.runData) render();
    }, 100);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    // Clear training timeout
    if (trainingTimeout) {
        clearTimeout(trainingTimeout);
    }
    // Close WebSocket cleanly
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
    // Clear resize timeout
    clearTimeout(resizeTimeout);
});
