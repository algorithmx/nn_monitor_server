# Time Travel History Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add historical time-travel navigation within a training run - users can pause live mode and browse through past steps, then return to live mode.

**Architecture:** Frontend-only changes - the backend already provides complete run data via `/api/v1/runs/{run_id}` and WebSocket `run_history` message. The frontend will maintain state for live/paused modes and a step navigation system.

**Tech Stack:** Vanilla JavaScript (no framework), HTML5, CSS3, Canvas API for pulse visualization

---

## Task 1: Add State Management for History Navigation

**Files:**
- Modify: `static/index.html:335-341` (state object)

**Step 1: Add new state properties**

Add these properties to the `state` object:
```javascript
const state = {
    // ... existing properties ...
    isLiveMode: true,           // true = showing live updates, false = paused on specific step
    selectedStep: null,         // step number being viewed (null = latest when liveMode)
    stepsList: [],              // sorted list of all available step numbers for current run
    lastSeenStep: null,         // track the last step we saw for smart resume
    isTrainingActive: true      // track if training is still receiving updates
};
```

**Step 2: Run the server to verify no errors**

Run: `python main.py`
Expected: Server starts without errors on port 8000

**Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: add state properties for history navigation"
```

---

## Task 2: Build History Control UI Components

**Files:**
- Modify: `static/index.html:35-75` (CSS - add new styles after header styles)
- Modify: `static/index.html:294-330` (HTML - add controls after header)

**Step 1: Add CSS styles for history controls**

Add these CSS rules after the `@keyframes pulse` block (around line 75):

```css
/* History controls section */
.history-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
}

/* Mode badge */
.mode-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 3px;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    background: var(--bg);
    border: 1px solid var(--border);
}

.mode-badge.live {
    background: rgba(74, 153, 137, 0.15);
    border-color: var(--health-good);
    color: var(--health-good);
}

.mode-badge.live .badge-dot {
    background: var(--health-good);
    animation: pulse 2s infinite;
}

.mode-badge.paused {
    background: rgba(200, 132, 68, 0.15);
    border-color: var(--health-warn);
    color: var(--health-warn);
}

.mode-badge.paused .badge-dot {
    background: var(--health-warn);
    animation: none;
}

.badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}

/* Live/Pause button */
.btn-live {
    padding: 6px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-bright);
    font-family: inherit;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    cursor: pointer;
    border-radius: 3px;
    transition: all 0.15s;
}

.btn-live:hover {
    border-color: #333;
}

.btn-live.live-active {
    background: rgba(74, 153, 137, 0.2);
    border-color: var(--health-good);
    color: var(--health-good);
    animation: subtle-pulse 2s infinite;
}

.btn-live.paused-active {
    background: rgba(200, 132, 68, 0.2);
    border-color: var(--health-warn);
    color: var(--health-warn);
}

@keyframes subtle-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(74, 153, 137, 0.4); }
    50% { box-shadow: 0 0 0 3px rgba(74, 153, 137, 0); }
}

/* Navigation controls */
.nav-controls {
    display: flex;
    align-items: center;
    gap: 8px;
}

.nav-btn {
    width: 28px;
    height: 28px;
    padding: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: inherit;
    font-size: 12px;
    cursor: pointer;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
}

.nav-btn:hover:not(:disabled) {
    border-color: #333;
    color: var(--text-bright);
}

.nav-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
}

/* Slider */
.step-slider-container {
    display: flex;
    align-items: center;
    gap: 8px;
}

.step-slider {
    flex: 1;
    min-width: 120px;
    height: 4px;
    -webkit-appearance: none;
    appearance: none;
    background: var(--border);
    outline: none;
    border-radius: 2px;
}

.step-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    background: var(--text-bright);
    cursor: pointer;
    border-radius: 50%;
    transition: all 0.15s;
}

.step-slider::-webkit-slider-thumb:hover {
    transform: scale(1.2);
    background: var(--health-good);
}

.step-slider::-moz-range-thumb {
    width: 12px;
    height: 12px;
    background: var(--text-bright);
    cursor: pointer;
    border: none;
    border-radius: 50%;
    transition: all 0.15s;
}

.step-slider::-moz-range-thumb:hover {
    transform: scale(1.2);
    background: var(--health-good);
}

.step-slider:disabled {
    opacity: 0.3;
    cursor: not-allowed;
}

.step-display {
    min-width: 60px;
    text-align: right;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    color: var(--text-bright);
}
```

**Step 2: Add HTML structure for history controls**

Add this section after the `<header>` tag (after line 300):

```html
<div class="history-controls">
    <div class="mode-badge live" id="modeBadge">
        <span class="badge-dot"></span>
        <span id="modeText">Live</span>
    </div>

    <button class="btn-live live-active" id="btnLivePause">⏸ Pause</button>

    <div class="nav-controls" id="navControls" style="opacity: 0.3; pointer-events: none;">
        <button class="nav-btn" id="btnFirst" title="First step">⏮</button>
        <button class="nav-btn" id="btnPrev" title="Previous step">◀</button>
        <button class="nav-btn" id="btnNext" title="Next step">▶</button>
        <button class="nav-btn" id="btnLast" title="Latest step">⏭</button>
    </div>

    <div class="step-slider-container">
        <input type="range" class="step-slider" id="stepSlider" min="0" max="100" value="0" disabled>
        <span class="step-display" id="stepDisplay">—</span>
    </div>
</div>
```

**Step 3: Refresh browser to verify UI appears**

Run: `python main.py` (if not running)
Expected: New controls section appears between header and main content, with Live badge, Pause button, and disabled nav controls

**Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: add history control UI components"
```

---

## Task 3: Implement Mode Toggle Logic

**Files:**
- Modify: `static/index.html:630` (before `connect()` call)

**Step 1: Add mode toggle functions**

Add these functions before `connect()`:

```javascript
// Toggle between live and paused mode
function toggleLiveMode() {
    state.isLiveMode = !state.isLiveMode;

    if (state.isLiveMode) {
        // Returning to live mode
        state.selectedStep = null;
        // Smart resume: check if training is still active
        if (!state.isTrainingActive && state.lastSeenStep !== null) {
            // Training ended, stay on last viewed step
            state.selectedStep = state.lastSeenStep;
        }
    } else {
        // Pausing - capture current step
        const steps = state.runData?.steps || [];
        if (steps.length > 0) {
            state.selectedStep = steps[steps.length - 1].step;
        }
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
        modeText.textContent = 'Live';
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
```

**Step 2: Wire up event listeners**

Add event listeners after the runSelect change listener (after line 426):

```javascript
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
```

**Step 3: Test button toggles state**

Refresh browser and click Pause/Live button
Expected: Badge changes between "Live" (green, pulsing) and "Paused @ step" (amber), button appearance changes

**Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: implement mode toggle logic"
```

---

## Task 4: Implement Step Navigation Functions

**Files:**
- Modify: `static/index.html` (after toggleLiveMode function)

**Step 1: Add navigation functions**

Add these functions after `updateHistoryUI()`:

```javascript
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
```

**Step 2: Call updateStepsList in addStep**

Modify the `addStep` function (around line 456) to call `updateStepsList()` before `render()`:

```javascript
function addStep(stepData) {
    if (!state.runData) {
        state.runData = { steps: [] };
    }

    const idx = state.runData.steps.findIndex(s => s.step === stepData.step);
    if (idx >= 0) {
        state.runData.steps[idx] = stepData;
    } else {
        state.runData.steps.push(stepData);
        state.runData.steps.sort((a, b) => a.step - b.step);
    }

    // Update per-layer history for pulse viz
    stepData.layers.forEach(layer => {
        if (!state.history[layer.layer_id]) {
            state.history[layer.layer_id] = [];
        }
        state.history[layer.layer_id].push({
            step: stepData.step,
            actStd: layer.intermediate_features.activation_std,
            gradNorm: layer.gradient_flow.gradient_l2_norm
        });
        if (state.history[layer.layer_id].length > CONFIG.historyLen) {
            state.history[layer.layer_id].shift();
        }
    });

    // Track training activity
    state.lastSeenStep = stepData.step;
    state.isTrainingActive = true;

    updateStepsList();  // ADD THIS LINE

    render();
}
```

**Step 3: Test navigation in paused mode**

1. Start server and test client
2. Wait for some data to arrive
3. Click Pause
4. Use nav buttons and slider
Expected: Can navigate between steps, UI updates to show current step position

**Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: implement step navigation functions"
```

---

## Task 5: Update render() for Historical Data Display

**Files:**
- Modify: `static/index.html:459-508` (render function)

**Step 1: Replace render() to support historical steps**

Replace the entire `render()` function with:

```javascript
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
    container.innerHTML = displayStep.layers.map(layer => {
        const health = assessHealth(layer);
        return `
            <div class="layer ${health.class}">
                <div class="layer-header">
                    <div>
                        <div class="layer-name">${layer.layer_id}</div>
                        <div class="layer-type">${layer.layer_type}</div>
                    </div>
                </div>
                <canvas class="pulse-viz" id="pulse-${layer.layer_id.replace(/[^a-z0-9]/gi, '-')}"></canvas>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value ${health.act}">${layer.intermediate_features.activation_std.toFixed(4)}</div>
                        <div class="metric-label">act std</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value ${health.grad}">${layer.gradient_flow.gradient_l2_norm.toFixed(4)}</div>
                        <div class="metric-label">grad norm</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value ${health.ratio}">${layer.intermediate_features.cross_layer_std_ratio?.toFixed(3) ?? '—'}</div>
                        <div class="metric-label">ratio</div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Draw pulse lines with historical context
    displayStep.layers.forEach(layer => {
        drawPulse(layer.layer_id, displayStep.step);
    });
}
```

**Step 2: Update drawPulse for historical context**

Replace the `drawPulse` function (around line 543) with:

```javascript
function drawPulse(layerId, currentStep) {
    const canvas = document.getElementById(`pulse-${layerId.replace(/[^a-z0-9]/gi, '-')}`);
    if (!canvas) return;

    // Get history leading up to current step
    const allHistory = state.history[layerId] || [];

    // Find the 50 steps leading up to and including currentStep
    let pulseHistory = [];
    if (allHistory.length > 0) {
        const currentStepIndex = allHistory.findIndex(h => h.step === currentStep);
        if (currentStepIndex >= 0) {
            // Show 50 steps leading up to current step
            const startIdx = Math.max(0, currentStepIndex - CONFIG.historyLen + 1);
            pulseHistory = allHistory.slice(startIdx, currentStepIndex + 1);
        } else {
            // Current step not in history yet, use available history
            pulseHistory = allHistory.slice(-CONFIG.historyLen);
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

    // Find min/max for scaling
    let maxVal = 0;
    pulseHistory.forEach(pt => {
        maxVal = Math.max(maxVal, pt.actStd, pt.gradNorm);
    });
    maxVal = Math.max(maxVal, 0.001);

    // Draw activation line
    ctx.strokeStyle = '#4a9';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    pulseHistory.forEach((pt, i) => {
        const x = (i / (pulseHistory.length - 1)) * w;
        const y = h - (pt.actStd / maxVal) * h * 0.8 - h * 0.1;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw gradient line
    ctx.strokeStyle = '#a371f7';
    ctx.lineWidth = 1;
    ctx.beginPath();

    pulseHistory.forEach((pt, i) => {
        const x = (i / (pulseHistory.length - 1)) * w;
        const y = h - (pt.gradNorm / maxVal) * h * 0.8 - h * 0.1;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
}
```

**Step 3: Test historical visualization**

1. Start server and test client
2. Let it run for ~30 steps
3. Pause and navigate to step 10
4. Observe pulse shows steps 1-10
5. Navigate to step 25
6. Observe pulse shows steps ~21-25

Expected: Pulse visualization shows 50 steps leading up to and including the selected step

**Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: update render for historical data display with context pulse"
```

---

## Task 6: Handle WebSocket Updates in History Mode

**Files:**
- Modify: `static/index.html:368-399` (handleMessage function)

**Step 1: Update handleMessage to respect history mode**

Modify the `new_metrics` case in `handleMessage`:

```javascript
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
        // Only add step and update display if in live mode
        // In paused mode, we still store the data but don't refresh display
        addStep(msg.data);

        // If in paused mode, we need to update the slider/steps list
        // but don't call render() to avoid disrupting the user's view
        if (!state.isLiveMode) {
            updateStepsList();
            updateHistoryUI(); // Update button states
        }
    }
    break;
```

**Step 2: Add training inactivity detection**

Add this new function after `updateStepsList()`:

```javascript
// Detect when training has ended (no new data for 10 seconds)
let trainingTimeout = null;
function resetTrainingTimeout() {
    if (trainingTimeout) clearTimeout(trainingTimeout);
    trainingTimeout = setTimeout(() => {
        state.isTrainingActive = false;
        if (state.isLiveMode) {
            updateHistoryUI();
        }
    }, 10000);
}
```

**Step 3: Call resetTrainingTimeout on new metrics**

Update the `new_metrics` case again to call `resetTrainingTimeout()`:

```javascript
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
        resetTrainingTimeout(); // ADD THIS
        addStep(msg.data);

        if (!state.isLiveMode) {
            updateStepsList();
            updateHistoryUI();
        }
    }
    break;
```

**Step 4: Test smart resume behavior**

1. Start server and test client
2. Wait for data, then pause
3. Navigate to a historical step
4. Stop test client (Ctrl+C)
5. Wait 10+ seconds for training timeout
6. Click Live button

Expected: Stays on the same step (training ended detection)

**Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: handle websocket updates in history mode with smart resume"
```

---

## Task 7: Update selectRun for Fresh Data

**Files:**
- Modify: `static/index.html:414-422` (selectRun function)

**Step 1: Reset history state on run change**

Modify `selectRun` function:

```javascript
function selectRun(id) {
    state.currentRun = id;
    document.getElementById('runSelect').value = id;
    state.history = {};

    // Reset history navigation state
    state.isLiveMode = true;
    state.selectedStep = null;
    state.stepsList = [];
    state.isTrainingActive = true;
    resetTrainingTimeout();

    updateHistoryUI();

    if (state.ws?.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ action: 'subscribe_run', run_id: id }));
    }
}
```

**Step 2: Test switching runs**

1. Start server with two different run IDs
2. Switch between runs in selector
3. Verify history controls reset to live mode

Expected: Switching runs resets to live mode with fresh data

**Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: reset history state on run change"
```

---

## Task 8: Final Testing and Polish

**Files:**
- Modify: `static/index.html` (if needed)

**Step 1: Run comprehensive test**

Test the full workflow:
1. Start server: `python main.py`
2. In another terminal: `python test_client.py --run-id test1 --steps 50 --interval 0.5`
3. Open browser to `http://localhost:8000`
4. Verify live mode shows live updates
5. Click Pause - verify badge changes to "Paused"
6. Use nav buttons to step through history
7. Use slider to jump to different steps
8. Click Live - verify returns to live mode
9. Pause again, navigate to step 10
10. Stop test client
11. Wait 10 seconds
12. Click Live - verify stays on step 10 (smart resume)
13. Start test_client again with different run_id
14. Switch runs in selector - verify resets to live

**Step 2: Check for edge cases**

- What happens with only 1 step? (nav buttons should be disabled)
- What happens when switching to a run with no data?
- What happens when WebSocket reconnects?

**Step 3: Fix any issues found**

If issues found, make targeted fixes

**Step 4: Final commit**

```bash
git add static/index.html
git commit -m "fix: address edge cases and polish"
```

---

## Task 9: Update Documentation

**Files:**
- Create: `docs/FEATURES.md` (or append to existing)

**Step 1: Document the history feature**

Create/update documentation:

```markdown
# Features

## Time Travel History Navigation

The monitor now supports browsing historical training data within a run.

### Usage

1. **Live Mode** (default): Shows real-time training updates with a pulsing green "● LIVE" badge
2. **Pause**: Click the "⏸ Pause" button to freeze the display and enter history mode
3. **Navigate**: Use navigation buttons or slider to browse historical steps:
   - ⏮ First step
   - ◀ Previous step
   - ▶ Next step
   - ⏭ Latest step
   - Slider: Drag to jump to any step
4. **Return to Live**: Click "▶ Live" to resume live updates

### Smart Resume

When training has ended (no new data for 10 seconds), returning to live mode will keep you on the last viewed step rather than jumping.

### Historical Pulse Visualization

When viewing a historical step, the pulse visualization shows the 50 steps leading up to and including that step, providing context for the trends.
```

**Step 2: Commit documentation**

```bash
git add docs/FEATURES.md
git commit -m "docs: document time travel history feature"
```

---

## Summary

This implementation adds time-travel history navigation without any backend changes. All state is maintained client-side, leveraging the existing `run_history` WebSocket message that provides complete run data.

### Key Files Modified
- `static/index.html` - Frontend application (only file changed)

### No Backend Changes Required
The backend already provides:
- Complete run history via `/api/v1/runs/{run_id}` and WebSocket `run_history`
- All step data in the response
- Real-time updates via `new_metrics` broadcasts

### Testing Checklist
- [ ] Live mode shows real-time updates
- [ ] Pause button switches to paused mode
- [ ] Nav buttons work correctly
- [ ] Slider allows quick navigation
- [ ] Pulse shows historical context (50 steps leading to selection)
- [ ] Smart resume works when training ends
- [ ] Switching runs resets to live mode
- [ ] Edge cases handled (1 step, empty runs, reconnect)
