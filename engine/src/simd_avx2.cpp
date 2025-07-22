#include "../include/nnue_engine.h"

#ifdef __AVX2__
#include <immintrin.h>
#include <algorithm>

namespace nnue {
namespace simd {

// AVX2 convolution implementation
void conv2d_unrolled_avx2(const float* input, const int8_t* weights,
                          const int32_t* biases, int8_t* output, float scale) {
    // For simplicity, fall back to scalar for conv layer
    // In practice, you'd want to optimize this with AVX2 as well
    conv2d_unrolled_scalar(input, weights, biases, output, scale);
}

// AVX2 feature transformer implementation
void ft_forward_avx2(const std::vector<int>& features, const int16_t* weights,
                     const int32_t* biases, int16_t* output, float scale) {
    const __m256i scale_vec = _mm256_set1_epi32(static_cast<int32_t>(scale));
    
    // Initialize with biases (process 8 int32s at a time)
    for (int i = 0; i < L1_SIZE; i += 8) {
        // Load 8 biases
        __m256i bias_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(biases + i));
        
        // Divide by scale and convert to int16
        __m256i scaled = _mm256_div_epi32(bias_vec, scale_vec);
        __m128i packed = _mm256_cvtepi32_epi16(scaled);
        
        // Store result
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), packed);
    }
    
    // Accumulate features (process 16 int16s at a time)
    for (int feature_idx : features) {
        const int16_t* feature_weights = weights + feature_idx * L1_SIZE;
        
        for (int i = 0; i < L1_SIZE; i += 16) {
            // Load current accumulator values
            __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(output + i));
            
            // Load feature weights
            __m256i weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(feature_weights + i));
            
            // Add weights to accumulator
            __m256i result = _mm256_add_epi16(acc, weights_vec);
            
            // Store result
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), result);
        }
    }
}

// AVX2 dense layer implementation
void dense_forward_avx2(const int16_t* input, const int8_t* weights,
                        const int32_t* biases, int8_t* output,
                        int input_size, int output_size, float scale) {
    
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        __m256i acc_vec = _mm256_set1_epi32(biases[out_idx]);
        
        // Process 16 inputs at a time (int16 x int8 -> int32)
        for (int in_idx = 0; in_idx < input_size; in_idx += 16) {
            int remaining = std::min(16, input_size - in_idx);
            
            if (remaining == 16) {
                // Load 16 int16 inputs
                __m256i input_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + in_idx));
                
                // Load 16 int8 weights and convert to int16
                __m128i weights_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + out_idx * input_size + in_idx));
                __m256i weights_vec = _mm256_cvtepi8_epi16(weights_i8);
                
                // Multiply and accumulate
                __m256i prod = _mm256_madd_epi16(input_vec, weights_vec);
                acc_vec = _mm256_add_epi32(acc_vec, prod);
            } else {
                // Handle remaining elements scalar
                int32_t scalar_acc = 0;
                for (int j = 0; j < remaining; ++j) {
                    int weight_idx = out_idx * input_size + in_idx + j;
                    int32_t prod = static_cast<int32_t>(input[in_idx + j]) * static_cast<int32_t>(weights[weight_idx]);
                    scalar_acc += prod;
                }
                // Add to accumulator
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_set1_epi32(scalar_acc));
            }
        }
        
        // Horizontal sum of accumulator
        __m128i acc_lo = _mm256_castsi256_si128(acc_vec);
        __m128i acc_hi = _mm256_extracti128_si256(acc_vec, 1);
        __m128i acc_sum = _mm_add_epi32(acc_lo, acc_hi);
        acc_sum = _mm_hadd_epi32(acc_sum, acc_sum);
        acc_sum = _mm_hadd_epi32(acc_sum, acc_sum);
        
        int32_t final_acc = _mm_extract_epi32(acc_sum, 0);
        
        // Apply scaling and clipped ReLU
        int32_t scaled = final_acc / static_cast<int32_t>(scale);
        output[out_idx] = static_cast<int8_t>(std::max(0, std::min(127, scaled)));
    }
}

} // namespace simd
} // namespace nnue

#else
// If AVX2 not available, provide empty implementations
namespace nnue {
namespace simd {

void conv2d_unrolled_avx2(const float* input, const int8_t* weights,
                          const int32_t* biases, int8_t* output, float scale) {
    conv2d_unrolled_scalar(input, weights, biases, output, scale);
}

void ft_forward_avx2(const std::vector<int>& features, const int16_t* weights,
                     const int32_t* biases, int16_t* output, float scale) {
    ft_forward_scalar(features, weights, biases, output, scale);
}

void dense_forward_avx2(const int16_t* input, const int8_t* weights,
                        const int32_t* biases, int8_t* output,
                        int input_size, int output_size, float scale) {
    dense_forward_scalar(input, weights, biases, output, input_size, output_size, scale);
}

} // namespace simd
} // namespace nnue

#endif 