#include "../include/nnue_engine.h"

#ifdef __AVX2__
#include <immintrin.h>
#include <algorithm>

namespace nnue {
namespace simd {

// AVX2 convolution implementation
void conv2d_unrolled_avx2(const float* input, const int8_t* weights,
                          const int32_t* biases, int8_t* output, float scale,
                          int input_h, int input_w, int out_channels, int stride) {
    // For simplicity, fall back to scalar for conv layer
    // In practice, you'd want to optimize this with AVX2 as well
    conv2d_unrolled_scalar(input, weights, biases, output, scale, input_h, input_w, out_channels, stride);
}

// AVX2 feature transformer implementation
void ft_forward_avx2(const std::vector<int>& features, const int16_t* weights,
                     const int32_t* biases, int16_t* output, int num_features,
                     int output_size, float scale) {
    // Initialize with biases
    for (int i = 0; i < output_size; ++i) {
        output[i] = static_cast<int16_t>(biases[i]);
    }
    
    // Accumulate features (process 16 int16s at a time with AVX2)
    for (int feature_idx : features) {
        if (feature_idx >= 0 && feature_idx < num_features) {
            const int16_t* feature_weights = weights + feature_idx * output_size;
            
            int i = 0;
            // Process 16 elements at a time
            for (; i <= output_size - 16; i += 16) {
                // Load current accumulator values
                __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(output + i));
                
                // Load feature weights
                __m256i weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(feature_weights + i));
                
                // Add weights to accumulator
                __m256i result = _mm256_add_epi16(acc, weights_vec);
                
                // Store result
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), result);
            }
            
            // Handle remaining elements
            for (; i < output_size; ++i) {
                output[i] += feature_weights[i];
            }
        }
    }
}

// Chess engine-style AVX2 optimized add feature
void add_feature_avx2(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size) {
    if (feature_idx < 0) return;
    
    const int16_t* feature_weights = weights + feature_idx * output_size;
    
    int i = 0;
    // Process 16 int16s at a time
    for (; i <= output_size - 16; i += 16) {
        // Load current accumulator values
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator + i));
        
        // Load feature weights
        __m256i weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(feature_weights + i));
        
        // Add weights to accumulator
        __m256i result = _mm256_add_epi16(acc, weights_vec);
        
        // Store result
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator + i), result);
    }
    
    // Handle remaining elements
    for (; i < output_size; ++i) {
        accumulator[i] += feature_weights[i];
    }
}

// Chess engine-style AVX2 optimized remove feature
void remove_feature_avx2(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size) {
    if (feature_idx < 0) return;
    
    const int16_t* feature_weights = weights + feature_idx * output_size;
    
    int i = 0;
    // Process 16 int16s at a time
    for (; i <= output_size - 16; i += 16) {
        // Load current accumulator values
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator + i));
        
        // Load feature weights
        __m256i weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(feature_weights + i));
        
        // Subtract weights from accumulator
        __m256i result = _mm256_sub_epi16(acc, weights_vec);
        
        // Store result
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator + i), result);
    }
    
    // Handle remaining elements
    for (; i < output_size; ++i) {
        accumulator[i] -= feature_weights[i];
    }
}

// AVX2 dense layer implementation
void dense_forward_avx2(const int16_t* input, const int8_t* weights,
                        const int32_t* biases, int8_t* output,
                        int input_size, int output_size, float scale) {
    
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        __m256i acc_vec = _mm256_set1_epi32(biases[out_idx]);
        
        // Process 16 inputs at a time (int16 x int8 -> int32)
        int in_idx = 0;
        for (; in_idx <= input_size - 16; in_idx += 16) {
            // Load 16 int16 inputs
            __m256i input_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + in_idx));
            
            // Load 16 int8 weights and convert to int16
            __m128i weights_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + out_idx * input_size + in_idx));
            __m256i weights_vec = _mm256_cvtepi8_epi16(weights_i8);
            
            // Multiply and accumulate
            __m256i prod = _mm256_madd_epi16(input_vec, weights_vec);
            acc_vec = _mm256_add_epi32(acc_vec, prod);
        }
        
        // Horizontal sum of acc_vec
        __m128i sum_128 = _mm_add_epi32(_mm256_castsi256_si128(acc_vec), _mm256_extracti128_si256(acc_vec, 1));
        sum_128 = _mm_add_epi32(sum_128, _mm_shuffle_epi32(sum_128, 0x4E));
        sum_128 = _mm_add_epi32(sum_128, _mm_shuffle_epi32(sum_128, 0xB1));
        int32_t sum = _mm_cvtsi128_si32(sum_128);
        
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
// If AVX2 not available, provide empty implementations
namespace nnue {
namespace simd {

void conv2d_unrolled_avx2(const float* input, const int8_t* weights,
                          const int32_t* biases, int8_t* output, float scale,
                          int input_h, int input_w, int out_channels, int stride) {
    conv2d_unrolled_scalar(input, weights, biases, output, scale, input_h, input_w, out_channels, stride);
}

void ft_forward_avx2(const std::vector<int>& features, const int16_t* weights,
                     const int32_t* biases, int16_t* output, int num_features,
                     int output_size, float scale) {
    ft_forward_scalar(features, weights, biases, output, num_features, output_size, scale);
}

void add_feature_avx2(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size) {
    add_feature_scalar(feature_idx, weights, accumulator, output_size);
}

void remove_feature_avx2(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size) {
    remove_feature_scalar(feature_idx, weights, accumulator, output_size);
}

void dense_forward_avx2(const int16_t* input, const int8_t* weights,
                        const int32_t* biases, int8_t* output,
                        int input_size, int output_size, float scale) {
    dense_forward_scalar(input, weights, biases, output, input_size, output_size, scale);
}

} // namespace simd
} // namespace nnue

#endif 