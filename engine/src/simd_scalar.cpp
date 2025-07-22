#include "../include/nnue_engine.h"
#include <algorithm>
#include <cmath>

namespace nnue {
namespace simd {

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
                            const int32_t* biases, int8_t* output, float scale) {
    // Unrolled convolution with stride=12 for 96x96x3 -> 8x8x12
    // Input: [96, 96, 3] in HWC format
    // Output: [8, 8, 12] in HWC format
    // Weights: [12, 3, 3, 3] in NCHW format
    
    constexpr int stride = 12;
    constexpr int padding = 1;
    
    for (int out_c = 0; out_c < OUTPUT_CHANNELS; ++out_c) {
        for (int out_h = 0; out_h < OUTPUT_GRID_SIZE; ++out_h) {
            for (int out_w = 0; out_w < OUTPUT_GRID_SIZE; ++out_w) {
                int32_t acc = biases[out_c];
                
                // Apply 3x3 kernel
                for (int kh = 0; kh < CONV_KERNEL_SIZE; ++kh) {
                    for (int kw = 0; kw < CONV_KERNEL_SIZE; ++kw) {
                        for (int in_c = 0; in_c < INPUT_CHANNELS; ++in_c) {
                            // Calculate input coordinates
                            int in_h = out_h * stride + kh - padding;
                            int in_w = out_w * stride + kw - padding;
                            
                            // Bounds check (zero padding)
                            if (in_h >= 0 && in_h < INPUT_IMAGE_SIZE && 
                                in_w >= 0 && in_w < INPUT_IMAGE_SIZE) {
                                
                                // Input index: HWC format
                                int input_idx = (in_h * INPUT_IMAGE_SIZE + in_w) * INPUT_CHANNELS + in_c;
                                
                                // Weight index: NCHW format
                                int weight_idx = ((out_c * INPUT_CHANNELS + in_c) * CONV_KERNEL_SIZE + kh) * CONV_KERNEL_SIZE + kw;
                                
                                acc += static_cast<int32_t>(input[input_idx] * scale) * static_cast<int32_t>(weights[weight_idx]);
                            }
                        }
                    }
                }
                
                // Apply quantization and clipping
                int32_t quantized = acc / static_cast<int32_t>(scale);
                output[(out_h * OUTPUT_GRID_SIZE + out_w) * OUTPUT_CHANNELS + out_c] = 
                    static_cast<int8_t>(std::max(-127, std::min(127, quantized)));
            }
        }
    }
}

// Scalar feature transformer implementation
void ft_forward_scalar(const std::vector<int>& features, const int16_t* weights,
                       const int32_t* biases, int16_t* output, float scale) {
    // Initialize with biases
    for (int i = 0; i < L1_SIZE; ++i) {
        output[i] = static_cast<int16_t>(biases[i] / static_cast<int32_t>(scale));
    }
    
    // Accumulate features
    for (int feature_idx : features) {
        const int16_t* feature_weights = weights + feature_idx * L1_SIZE;
        for (int i = 0; i < L1_SIZE; ++i) {
            output[i] += feature_weights[i];
        }
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
        int32_t scaled = acc / static_cast<int32_t>(scale);
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