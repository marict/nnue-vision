#include "../include/nnue_engine.h"

#ifdef __ARM_NEON__
#include <arm_neon.h>
#include <algorithm>

namespace nnue {
namespace simd {

// NEON convolution implementation
void conv2d_unrolled_neon(const float* input, const int8_t* weights,
                          const int32_t* biases, int8_t* output, float scale,
                          int input_h, int input_w, int out_channels, int stride) {
    // For simplicity, fall back to scalar for conv layer
    // In practice, you'd want to optimize this with NEON as well
    conv2d_unrolled_scalar(input, weights, biases, output, scale, input_h, input_w, out_channels, stride);
}

// NEON feature transformer implementation
void ft_forward_neon(const std::vector<int>& features, const int16_t* weights,
                     const int32_t* biases, int16_t* output, int num_features,
                     int output_size, float scale) {
    // Initialize with biases
    for (int i = 0; i < output_size; ++i) {
        output[i] = static_cast<int16_t>(biases[i]);
    }
    
    // Accumulate features (process 8 int16s at a time with NEON)
    for (int feature_idx : features) {
        if (feature_idx >= 0 && feature_idx < num_features) {
            const int16_t* feature_weights = weights + feature_idx * output_size;
            
            int i = 0;
            // Process 8 elements at a time
            for (; i <= output_size - 8; i += 8) {
                // Load current accumulator values
                int16x8_t acc = vld1q_s16(output + i);
                
                // Load feature weights
                int16x8_t weights_vec = vld1q_s16(feature_weights + i);
                
                // Add weights to accumulator
                int16x8_t result = vaddq_s16(acc, weights_vec);
                
                // Store result
                vst1q_s16(output + i, result);
            }
            
            // Handle remaining elements
            for (; i < output_size; ++i) {
                output[i] += feature_weights[i];
            }
        }
    }
}

// Chess engine-style NEON optimized add feature
void add_feature_neon(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size) {
    if (feature_idx < 0) return;
    
    const int16_t* feature_weights = weights + feature_idx * output_size;
    
    int i = 0;
    // Process 8 int16s at a time with NEON
    for (; i <= output_size - 8; i += 8) {
        // Load current accumulator values
        int16x8_t acc = vld1q_s16(accumulator + i);
        
        // Load feature weights
        int16x8_t weights_vec = vld1q_s16(feature_weights + i);
        
        // Add weights to accumulator
        int16x8_t result = vaddq_s16(acc, weights_vec);
        
        // Store result
        vst1q_s16(accumulator + i, result);
    }
    
    // Handle remaining elements
    for (; i < output_size; ++i) {
        accumulator[i] += feature_weights[i];
    }
}

// Chess engine-style NEON optimized remove feature
void remove_feature_neon(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size) {
    if (feature_idx < 0) return;
    
    const int16_t* feature_weights = weights + feature_idx * output_size;
    
    int i = 0;
    // Process 8 int16s at a time with NEON
    for (; i <= output_size - 8; i += 8) {
        // Load current accumulator values
        int16x8_t acc = vld1q_s16(accumulator + i);
        
        // Load feature weights
        int16x8_t weights_vec = vld1q_s16(feature_weights + i);
        
        // Subtract weights from accumulator
        int16x8_t result = vsubq_s16(acc, weights_vec);
        
        // Store result
        vst1q_s16(accumulator + i, result);
    }
    
    // Handle remaining elements
    for (; i < output_size; ++i) {
        accumulator[i] -= feature_weights[i];
    }
}

// NEON dense layer implementation
void dense_forward_neon(const int16_t* input, const int8_t* weights,
                        const int32_t* biases, int8_t* output,
                        int input_size, int output_size, float scale) {
    
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        int32x4_t acc_vec = vdupq_n_s32(biases[out_idx]);
        
        // Process 8 inputs at a time (int16 x int8 -> int32)
        int in_idx = 0;
        for (; in_idx <= input_size - 8; in_idx += 8) {
            // Load 8 int16 inputs
            int16x8_t input_vec = vld1q_s16(input + in_idx);
            
            // Load 8 int8 weights and convert to int16
            int8x8_t weights_i8 = vld1_s8(weights + out_idx * input_size + in_idx);
            int16x8_t weights_vec = vmovl_s8(weights_i8);
            
            // Multiply and accumulate
            int32x4_t prod_lo = vmull_s16(vget_low_s16(input_vec), vget_low_s16(weights_vec));
            int32x4_t prod_hi = vmull_s16(vget_high_s16(input_vec), vget_high_s16(weights_vec));
            
            acc_vec = vaddq_s32(acc_vec, prod_lo);
            acc_vec = vaddq_s32(acc_vec, prod_hi);
        }
        
        // Horizontal sum of accumulator vector
        int32x2_t sum_pair = vadd_s32(vget_low_s32(acc_vec), vget_high_s32(acc_vec));
        int32_t sum = vget_lane_s32(vpadd_s32(sum_pair, sum_pair), 0);
        
        // Handle remaining elements
        for (; in_idx < input_size; ++in_idx) {
            int weight_idx = out_idx * input_size + in_idx;
            sum += static_cast<int32_t>(input[in_idx]) * static_cast<int32_t>(weights[weight_idx]);
        }
        
        // Apply scaling and clipped ReLU
        int32_t scaled = sum / static_cast<int32_t>(scale);
        output[out_idx] = static_cast<int8_t>(std::max(0, std::min(127, scaled)));
    }
}

} // namespace simd
} // namespace nnue

#else
// If NEON not available, provide empty implementations
namespace nnue {
namespace simd {

void conv2d_unrolled_neon(const float* input, const int8_t* weights,
                          const int32_t* biases, int8_t* output, float scale,
                          int input_h, int input_w, int out_channels, int stride) {
    conv2d_unrolled_scalar(input, weights, biases, output, scale, input_h, input_w, out_channels, stride);
}

void ft_forward_neon(const std::vector<int>& features, const int16_t* weights,
                     const int32_t* biases, int16_t* output, int num_features,
                     int output_size, float scale) {
    ft_forward_scalar(features, weights, biases, output, num_features, output_size, scale);
}

void dense_forward_neon(const int16_t* input, const int8_t* weights,
                        const int32_t* biases, int8_t* output,
                        int input_size, int output_size, float scale) {
    dense_forward_scalar(input, weights, biases, output, input_size, output_size, scale);
}

} // namespace simd
} // namespace nnue

#endif 