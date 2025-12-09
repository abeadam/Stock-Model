# Alternatives to nn.Linear in PyTorch

## Common Alternatives for Your Use Case

### 1. **nn.Linear** (Current - Standard Fully Connected)
```python
self.fc3 = nn.Linear(64, 32)
```
- **Use when**: Standard dense layer needed
- **Pros**: Simple, efficient, well-optimized
- **Cons**: No built-in regularization

### 2. **nn.Conv1d** (1D Convolution)
```python
self.conv = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
# Reshape: x = x.unsqueeze(1)  # (batch, 1, features)
# After: x = x.squeeze(1)  # (batch, features)
```
- **Use when**: You want to learn local patterns or share weights
- **Pros**: Can capture local dependencies, parameter sharing
- **Cons**: Requires reshaping for 1D data

### 3. **nn.Bilinear** (Bilinear Transformation)
```python
self.bilinear = nn.Bilinear(in1_features=64, in2_features=64, out_features=32)
# Takes two inputs: out = bilinear(x1, x2)
```
- **Use when**: You need to combine two different feature vectors
- **Pros**: Captures interactions between two inputs
- **Cons**: Requires two inputs, more parameters

### 4. **nn.LazyLinear** (Lazy Linear - Infers Input Size)
```python
self.fc = nn.LazyLinear(32)  # Automatically infers input size
```
- **Use when**: Input size is unknown at initialization
- **Pros**: Convenient, no need to specify input size
- **Cons**: First forward pass is slower (lazy initialization)

### 5. **Custom Weighted Linear** (With Custom Initialization)
```python
class WeightedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Custom initialization
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```
- **Use when**: You need custom initialization or behavior
- **Pros**: Full control over initialization and behavior
- **Cons**: More code, need to handle everything manually

### 6. **nn.Sequential with Multiple Operations**
```python
self.fc_block = nn.Sequential(
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.1)
)
```
- **Use when**: You want a complete block (linear + norm + activation)
- **Pros**: Clean, modular code
- **Cons**: Less flexible than separate layers

### 7. **Functional API** (torch.nn.functional.linear)
```python
import torch.nn.functional as F

# In forward:
out = F.linear(x, self.weight, self.bias)
```
- **Use when**: You need dynamic behavior or custom logic
- **Pros**: More flexible, can change weights dynamically
- **Cons**: Need to manage parameters manually

### 8. **nn.MultiheadAttention** (For Feature Interactions)
```python
self.attention = nn.MultiheadAttention(
    embed_dim=64, 
    num_heads=4, 
    batch_first=True
)
# out, _ = attention(x, x, x)  # Self-attention
```
- **Use when**: You want attention mechanism in output layers
- **Pros**: Can learn important feature interactions
- **Cons**: More complex, requires reshaping

### 9. **Residual Connection with Linear**
```python
class ResidualLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.linear(x))
```
- **Use when**: Deep networks, helps with gradient flow
- **Pros**: Better gradient flow, can go deeper
- **Cons**: Slightly more complex

### 10. **nn.LayerNorm + Linear** (Normalization Before Linear)
```python
self.norm = nn.LayerNorm(64)
self.fc = nn.Linear(64, 32)

# In forward:
x = self.norm(x)
x = self.fc(x)
```
- **Use when**: You want normalized inputs to linear layer
- **Pros**: Can stabilize training
- **Cons**: Extra normalization step

## Recommendations for Your Model

For your **TransformerPredictor** output layers, consider:

1. **Keep nn.Linear** - It's the standard choice
2. **Add Residual Connections** - If you have many layers
3. **Use LayerNorm** - Instead of or in addition to BatchNorm
4. **Consider Attention** - For feature interactions in output layers

## Example: Enhanced Output Layer

```python
# Instead of just:
self.fc3 = nn.Linear(64, 32)

# You could use:
self.fc3_block = nn.Sequential(
    nn.LayerNorm(64),  # Normalize before linear
    nn.Linear(64, 32),
    nn.LayerNorm(32),  # Normalize after linear
    nn.GELU(),
    nn.Dropout(0.1)
)

# Or with residual:
class ResidualFC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x):
        return self.norm(x + self.linear(x))
```

