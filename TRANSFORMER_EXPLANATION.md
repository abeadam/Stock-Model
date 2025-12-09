# Transformer Model for SPX Prediction

## How Transformers Maintain Information from Previous Iterations

### Key Advantage Over LSTM

**LSTM**: Processes sequences sequentially, maintaining information in hidden states that can degrade over long sequences.

**Transformer**: Uses **self-attention** to directly attend to ALL previous timesteps simultaneously, maintaining information from every position in the sequence.

### How It Works

1. **Input Projection**: Projects input features to a model dimension (d_model)

2. **Positional Encoding**: Adds positional information so the model understands the order of timesteps
   - Uses sinusoidal encoding to represent position
   - Allows the model to distinguish between timestep 1 and timestep 20

3. **Self-Attention Mechanism**: 
   - Each timestep can attend to ALL other timesteps in the sequence
   - Attention weights determine how much each timestep should focus on each other timestep
   - This means timestep 20 can directly access information from timestep 1, 5, 10, etc.

4. **Multi-Head Attention**: 
   - Uses multiple attention heads (default: 8)
   - Each head can learn different relationships between timesteps
   - One head might focus on short-term patterns, another on long-term trends

5. **Feed-Forward Networks**: 
   - Processes the attended information through fully connected layers
   - Adds non-linearity to the model

6. **Layer Stacking**: 
   - Multiple transformer layers (default: 3) allow the model to learn increasingly complex patterns
   - Each layer refines the representations from the previous layer

### Example

For a sequence of 20 timesteps predicting SPX High/Low:

- **LSTM**: Information from timestep 1 flows through hidden states → timestep 2 → ... → timestep 20
  - Information can be lost or diluted through this chain
  
- **Transformer**: Timestep 20 can directly attend to:
  - Timestep 1 (earliest data)
  - Timestep 10 (mid-sequence)
  - Timestep 19 (immediately previous)
  - All other timesteps with learned attention weights

### Configuration

In `spx_prediction_model.py`, set:
```python
MODEL_TYPE = 'transformer'  # Use transformer
# or
MODEL_TYPE = 'lstm'  # Use LSTM
```

### Transformer Parameters

- `d_model=128`: Model dimension (size of feature vectors)
- `nhead=8`: Number of attention heads
- `num_layers=3`: Number of transformer encoder layers
- `dim_feedforward=512`: Size of feedforward network
- `max_seq_length=100`: Maximum sequence length for positional encoding

### When to Use Transformer vs LSTM

**Use Transformer when:**
- You need to capture long-range dependencies
- All timesteps in the sequence are equally important
- You want the model to learn which timesteps are most relevant

**Use LSTM when:**
- Sequential processing is more natural
- Recent timesteps are more important than distant ones
- You need faster training/inference

