# Debugging Setup

The virtual environment and IDE configuration have been set up for debugging.

## Virtual Environment

- **Location**: `./venv/`
- **Python**: 3.12
- **Packages**: All required packages are installed (torch, pandas, numpy, scikit-learn)

## Debugging in VS Code/Cursor

1. **Select Python Interpreter**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Choose `./venv/bin/python`

2. **Start Debugging**:
   - Press `F5` or go to Run → Start Debugging
   - Select "Python: SPX Prediction Model" from the debug configurations
   - Or use "Python: Current File" to debug any Python file

3. **Set Breakpoints**:
   - Click in the left margin next to line numbers to set breakpoints
   - The debugger will pause execution at breakpoints

## Running Without Debugging

```bash
source venv/bin/activate
python spx_prediction_model.py
```

## Verify Setup

```bash
source venv/bin/activate
python -c "import pandas, torch, sklearn; print('All packages available!')"
```

