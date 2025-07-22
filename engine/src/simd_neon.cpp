#include "../include/nnue_engine.h"

#ifdef __ARM_NEON__
#include <arm_neon.h>
#include <algorithm>

namespace nnue {
namespace simd {

// NEON convolution implementation
void conv2d_unrolled_neon(const float* input, const int8_t* weights,
                          const int32_t* biases, int8_t* output, float scale) {
    // For simplicity, fall back to scalar for conv layer
    // In practice, you'd want to optimize this with NEON as well
    conv2d_unrolled_scalar(input, weights, biases, output, scale);
}

// NEON feature transformer implementation
void ft_forward_neon(const std::vector<int>& features, const int16_t* weights,
                     const int32_t* biases, int16_t* output, float scale) {
    const int32_t scale_int = static_cast<int32_t>(scale);
    
    // Initialize with biases (process 4 int32s at a time)
    for (int i = 0; i < L1_SIZE; i += 4) {
        int remaining = std::min(4, L1_SIZE - i);
        
        if (remaining == 4) {
            // Load 4 biases and divide by scale using scalar division
            // NEON doesn't have integer division
            int16_t scaled_vals[4];
            for (int j = 0; j < 4; ++j) {
                scaled_vals[j] = static_cast<int16_t>(biases[i + j] / scale_int);
            }
            
            // Load as int16x4_t and store
            int16x4_t packed = vld1_s16(scaled_vals);
            vst1_s16(output + i, packed);
        } else {
            // Handle remaining elements scalar
            for (int j = 0; j < remaining; ++j) {
                output[i + j] = static_cast<int16_t>(biases[i + j] / scale_int);
            }
        }
    }
    
    // Accumulate features (process 8 int16s at a time)
    for (int feature_idx : features) {
        const int16_t* feature_weights = weights + feature_idx * L1_SIZE;
        
        for (int i = 0; i < L1_SIZE; i += 8) {
            int remaining = std::min(8, L1_SIZE - i);
            
            if (remaining == 8) {
                // Load current accumulator values
                int16x8_t acc = vld1q_s16(output + i);
                
                // Load feature weights
                int16x8_t weights_vec = vld1q_s16(feature_weights + i);
                
                // Add weights to accumulator
                int16x8_t result = vaddq_s16(acc, weights_vec);
                
                // Store result
                vst1q_s16(output + i, result);
            } else {
                // Handle remaining elements scalar
                for (int j = 0; j < remaining; ++j) {
                    output[i + j] += feature_weights[i + j];
                }
            }
        }
    }
}

// NEON dense layer implementation
void dense_forward_neon(const int16_t* input, const int8_t* weights,
                        const int32_t* biases, int8_t* output,
                        int input_size, int output_size, float scale) {
    
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        int32x4_t acc_vec = vdupq_n_s32(biases[out_idx]);
        
        // Process 8 inputs at a time (int16 x int8 -> int32)
        for (int in_idx = 0; in_idx < input_size; in_idx += 8) {
            int remaining = std::min(8, input_size - in_idx);
            
            if (remaining == 8) {
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
            } else {
                // Handle remaining elements scalar
                int32_t scalar_acc = 0;
                for (int j = 0; j < remaining; ++j) {
                    int weight_idx = out_idx * input_size + in_idx + j;
                    int32_t prod = static_cast<int32_t>(input[in_idx + j]) * static_cast<int32_t>(weights[weight_idx]);
                    scalar_acc += prod;
                }
                // Add to accumulator vector
                int32_t acc_vals[4];
                vst1q_s32(acc_vals, acc_vec);
                acc_vals[0] += scalar_acc;
                acc_vec = vld1q_s32(acc_vals);
            }
        }
        
        // Horizontal sum of accumulator
        int32x2_t acc_sum = vadd_s32(vget_low_s32(acc_vec), vget_high_s32(acc_vec));
        acc_sum = vpadd_s32(acc_sum, acc_sum);
        int32_t final_acc = vget_lane_s32(acc_sum, 0);
        
        // Apply scaling and clipped ReLU
        int32_t scaled = final_acc / static_cast<int32_t>(scale);
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
                          const int32_t* biases, int8_t* output, float scale) {
    conv2d_unrolled_scalar(input, weights, biases, output, scale);
}

void ft_forward_neon(const std::vector<int>& features, const int16_t* weights,
                     const int32_t* biases, int16_t* output, float scale) {
    ft_forward_scalar(features, weights, biases, output, scale);
}

void dense_forward_neon(const int16_t* input, const int8_t* weights,
                        const int32_t* biases, int8_t* output,
                        int input_size, int output_size, float scale) {
    dense_forward_scalar(input, weights, biases, output, input_size, output_size, scale);
}

} // namespace simd
} // namespace nnue

#endif 