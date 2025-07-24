#include "../include/nnue_engine.h"
#include <algorithm>
#include <climits>

namespace nnue {
namespace simd {

// Constants for configurable architecture
// These will be set dynamically based on loaded model
static int L1_SIZE = 256;  // Will be updated when model loads
static int L2_SIZE = 16;
static int L3_SIZE = 32;

// Utility function to set architecture constants
void set_architecture_constants(int l1_size, int l2_size, int l3_size) {
    L1_SIZE = l1_size;
    L2_SIZE = l2_size;
    L3_SIZE = l3_size;
}

// CPU capability detection
bool has_avx2() {
    #ifdef __AVX2__
        return true;
    #else
        return false;
    #endif
}

bool has_neon() {
    #ifdef __ARM_NEON__
        return true;
    #else
        return false;
    #endif
}

// Scalar convolution implementation
void conv2d_unrolled_scalar(const float* input, const int8_t* weights,
                           const int32_t* biases, int8_t* output, float scale,
                           int input_h, int input_w, int out_channels, int stride) {
    int kernel_h = 3, kernel_w = 3;  // Fixed 3x3 kernel
    int output_h = (input_h + 2 - kernel_h) / stride + 1;  // With padding=1
    int output_w = (input_w + 2 - kernel_w) / stride + 1;
    
    for (int out_c = 0; out_c < out_channels; ++out_c) {
        for (int out_h = 0; out_h < output_h; ++out_h) {
            for (int out_w = 0; out_w < output_w; ++out_w) {
                int32_t acc = biases[out_c];
                
                // Convolution with padding
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_h = out_h * stride + kh - 1;  // padding=1
                        int in_w = out_w * stride + kw - 1;
                        
                        if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                            for (int in_c = 0; in_c < 3; ++in_c) {  // RGB channels
                                int input_idx = (in_h * input_w + in_w) * 3 + in_c;
                                int weight_idx = ((out_c * kernel_h + kh) * kernel_w + kw) * 3 + in_c;
                                
                                acc += static_cast<int32_t>(input[input_idx] * scale) * weights[weight_idx];
                            }
                        }
                    }
                }
                
                // Apply activation and quantize
                int8_t result = static_cast<int8_t>(std::max(-127, std::min(127, acc / static_cast<int32_t>(scale))));
                int output_idx = (out_h * output_w + out_w) * out_channels + out_c;
                output[output_idx] = result;
            }
        }
    }
}

// Scalar feature transformer implementation
void ft_forward_scalar(const std::vector<int>& features, const int16_t* weights,
                       const int32_t* biases, int16_t* output, int num_features,
                       int output_size, float /* scale */) {
    // Initialize with biases
    for (int i = 0; i < output_size; ++i) {
        output[i] = static_cast<int16_t>(biases[i]);
    }
    
    // Accumulate features
    for (int feature_idx : features) {
        if (feature_idx >= 0 && feature_idx < num_features) {
            const int16_t* feature_weights = weights + feature_idx * output_size;
            for (int i = 0; i < output_size; ++i) {
                output[i] += feature_weights[i];
            }
        }
    }
}

// Chess engine-style incremental feature operations (scalar)
void add_feature_scalar(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size) {
    if (feature_idx < 0) return;
    
    const int16_t* feature_weights = weights + feature_idx * output_size;
    for (int i = 0; i < output_size; ++i) {
        accumulator[i] += feature_weights[i];
    }
}

void remove_feature_scalar(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size) {
    if (feature_idx < 0) return;
    
    const int16_t* feature_weights = weights + feature_idx * output_size;
    for (int i = 0; i < output_size; ++i) {
        accumulator[i] -= feature_weights[i];
    }
}

// Scalar dense layer implementation  
void dense_forward_scalar(const int16_t* input, const int8_t* weights,
                          const int32_t* biases, int8_t* output,
                          int input_size, int output_size, float scale) {
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        int32_t acc = biases[out_idx];
        
        // Matrix-vector multiplication
        for (int in_idx = 0; in_idx < input_size; ++in_idx) {
            int weight_idx = out_idx * input_size + in_idx;
            acc += static_cast<int32_t>(input[in_idx]) * static_cast<int32_t>(weights[weight_idx]);
        }
        
        // Apply scaling and clipped ReLU
        // For quantized arithmetic: result = acc / scale, but account for quantization
        // acc is sum of (quantized_input * quantized_weight), need to dequantize properly
        float result = static_cast<float>(acc) / scale;
        int32_t scaled = static_cast<int32_t>(result);
        output[out_idx] = static_cast<int8_t>(std::max(0, std::min(127, scaled)));
    }
}

// For int16 input version (used for L2 layer)
void dense_forward_scalar_int16(const int8_t* input, const int8_t* weights,
                                const int32_t* biases, int8_t* output,
                                int input_size, int output_size, float scale) {
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        int32_t acc = biases[out_idx];
        
        // Matrix-vector multiplication
        for (int in_idx = 0; in_idx < input_size; ++in_idx) {
            int weight_idx = out_idx * input_size + in_idx;
            acc += static_cast<int32_t>(input[in_idx]) * static_cast<int32_t>(weights[weight_idx]);
        }
        
        // Apply scaling and clipped ReLU
        int32_t scaled = acc / static_cast<int32_t>(scale);
        output[out_idx] = static_cast<int8_t>(std::max(0, std::min(127, scaled)));
    }
}

} // namespace simd
} // namespace nnue 