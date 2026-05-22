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

When training has ended (no new data for 10 seconds), returning to live mode will keep you on the last viewed step rather than jumping. The mode badge will show "Live (ended)" to indicate training has completed.

### Historical Pulse Visualization

When viewing a historical step, the pulse visualization shows the 50 steps leading up to and including that step, providing context for the trends.

### Technical Details

- All state is maintained client-side (no backend changes required)
- History is preserved until switching runs or refreshing the page
- Navigation controls are automatically disabled in live mode
- Alerts are hidden when viewing historical data

### WebSocket Message Flow

The time travel feature leverages these WebSocket messages:

1. **`run_history`**: Received when subscribing to a run, contains all historical steps
2. **`new_metrics`**: Real-time updates, stored in background when paused
3. **`initial_runs`**: List of available runs on connection

The frontend maintains these state variables:
- `isLiveMode`: Whether showing live updates or paused on a specific step
- `selectedStep`: The step number being viewed (null = latest in live mode)
- `stepsList`: Sorted list of all available step numbers
- `lastSeenStep`: The most recent step received
- `isTrainingActive`: Whether training is still receiving updates (10s timeout)
