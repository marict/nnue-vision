# NNUE Quantization Implementation

This repository contains implementations of NNUE (Efficiently Updatable Neural Networks) quantization schemes, including a **faithful implementation** that closely matches the original paper specification.

## 🎯 Faithful NNUE Implementation

The `original_nnue_quantization.py` contains a faithful implementation of the NNUE quantization scheme from the original paper:

> "Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi" by Yu Nasu (2018)

### ✨ Key Features Implemented

#### 1. **Per-Row Weight Scaling**
- One scale factor per output neuron (not global scaling)
- Better quantization precision: `scale = qmax / row_max`
- Separate scales for weights and biases: `bias_scale = weight_scale * input_scale`

#### 2. **Cached Quantized Weights**
- Weights quantized once per optimizer step (not every forward pass)
- Stored as `int16` for feature transformer, `int8` for other layers
- Significant performance improvement during inference

#### 3. **Proper Difference Calculation**
- Maintains per-sample accumulator of `W₁ × x` in int32
- Tracks previous input to compute differences
- Updates only changed feature indices: `O(changed_features)` vs `O(all_features)`
- Supports both stream mode (batch=1) and mini-batch mode

#### 4. **Integer Arithmetic Simulation**
- Feature transformer: `int16 × int16 → int32`
- Later layers: `int8 × int8 → int32`
- Minimizes floating-point operations until final output

#### 5. **Architectural Improvements**
- Removed unnecessary "two perspectives" concatenation (vision domain)
- Layer2 input: `feature_size` (not `2*feature_size`)
- ClippedReLU with threshold 127 (matches original)

### 🚀 API Surface

```python
# Create model
model = OriginalNNUEModel(input_features=768, feature_size=256, batch_size=1)

# Prepare quantized weights (call after optimizer step)
model.prepare_quant()

# Forward pass with difference calculation
output = model(x, use_difference_calc=True)

# Reset accumulator state
model.reset_state(batch_size=2)

# Get quantization information
info = model.get_quantization_info()
```

### 📊 Performance Benefits

1. **Difference Updates**: Only process changed features instead of all features
2. **Cached Weights**: No quantization overhead per forward pass
3. **Integer Math**: SIMD-friendly operations (simulated)
4. **Memory Efficient**: Persistent accumulator state

### 🔬 Architecture Comparison

| Component | Original Paper | My Implementation | Status |
|-----------|---------------|-------------------|---------|
| W₁ quantization | 16-bit integers | 16-bit integers | ✅ |
| W₂,W₃,W₄ quantization | 8-bit integers | 8-bit integers | ✅ |
| Per-row scaling | ✅ | ✅ | ✅ |
| Cached weights | ✅ | ✅ | ✅ |
| Difference calculation | ✅ | ✅ | ✅ |
| Two perspectives | ✅ (chess) | ❌ (vision) | Domain adapted |
| ClippedReLU(127) | ✅ | ✅ | ✅ |
| Integer accumulator | int32 | int32 simulation | ✅ |

### 🧪 Testing

Run the implementation to see all features in action:

```bash
python original_nnue_quantization.py
```

The test demonstrates:
- Full computation vs difference calculation
- State management and reset
- Stream mode (batch=1) and mini-batch mode
- Training vs inference modes
- Performance metrics and caching

### 📈 Results

The faithful implementation achieves:
- ✅ **Algorithmic fidelity** to the original NNUE paper
- ✅ **Performance optimizations** through difference calculation
- ✅ **Memory efficiency** with persistent state
- ✅ **Quantization accuracy** with per-row scaling
- ✅ **Training compatibility** with PyTorch fake quantization

This implementation bridges the gap between the original NNUE paper and modern PyTorch-based model compression techniques, providing both computational efficiency and model size reduction. 