#include "../include/nnue_engine.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace nnue {

ConvLayer::ConvLayer() : scale(64.0f), out_channels(0), in_channels(0), kernel_h(0), kernel_w(0) {}

bool ConvLayer::load_from_stream(std::ifstream& file) {
    // Read layer type (required by serialization format)
    uint32_t layer_type;
    file.read(reinterpret_cast<char*>(&layer_type), sizeof(uint32_t));
    
    // Read scale
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    
    // Read dimensions
    file.read(reinterpret_cast<char*>(&out_channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&in_channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&kernel_h), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&kernel_w), sizeof(uint32_t));
    
    if (in_channels != 3) {
        std::cerr << "Invalid conv input channels: " << in_channels << " (expected 3)" << std::endl;
        return false;
    }
    
    if (kernel_h != 3 || kernel_w != 3) {
        std::cerr << "Invalid conv kernel size: " << kernel_h << "x" << kernel_w << " (expected 3x3)" << std::endl;
        return false;
    }
    
    // Read weights (int8_t)
    long long weight_count = static_cast<long long>(out_channels) * in_channels * kernel_h * kernel_w;
    if (out_channels <= 0 || in_channels != 3 || kernel_h != 3 || kernel_w != 3 || weight_count <= 0) {
        std::cerr << "Invalid conv dimensions while loading weights" << std::endl;
        return false;
    }
    weights.resize(static_cast<size_t>(weight_count));
    file.read(reinterpret_cast<char*>(weights.data()), static_cast<std::streamsize>(weight_count));
    if (!file.good()) {
        std::cerr << "Failed to read conv weights" << std::endl;
        return false;
    }
    
    // Read bias dimensions and data
    uint32_t bias_count;
    file.read(reinterpret_cast<char*>(&bias_count), sizeof(uint32_t));
    if (bias_count != static_cast<uint32_t>(out_channels)) {
        std::cerr << "Conv bias count mismatch" << std::endl;
        return false;
    }
    
    biases.resize(bias_count);
    file.read(reinterpret_cast<char*>(biases.data()), bias_count * sizeof(int32_t));
    if (!file.good()) {
        std::cerr << "Failed to read conv biases" << std::endl;
        return false;
    }
    
    return true;
}

void ConvLayer::forward(const float* input, int8_t* output, int input_h, int input_w, int stride) const {
    int output_h = (input_h + 2 - kernel_h) / stride + 1;
    int output_w = (input_w + 2 - kernel_w) / stride + 1;
    
    const int interior_start_h = 1;
    const int interior_end_h = output_h - 1;
    const int interior_start_w = 1; 
    const int interior_end_w = output_w - 1;
    
    for (int out_c = 0; out_c < out_channels; ++out_c) {
        int32_t bias = biases[static_cast<size_t>(out_c)];
        
        for (int out_h = interior_start_h; out_h < interior_end_h; ++out_h) {
            for (int out_w = interior_start_w; out_w < interior_end_w; ++out_w) {
                int32_t acc = bias;
                
                const int base_in_h = out_h * stride - 1;
                const int base_in_w = out_w * stride - 1;
                
                for (int in_c = 0; in_c < in_channels; ++in_c) {
                    acc += static_cast<int32_t>(input[(base_in_h * input_w + base_in_w) * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 0) * 3 + 0) * in_channels + in_c];
                    acc += static_cast<int32_t>(input[(base_in_h * input_w + base_in_w + 1) * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 0) * 3 + 1) * in_channels + in_c];
                    acc += static_cast<int32_t>(input[(base_in_h * input_w + base_in_w + 2) * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 0) * 3 + 2) * in_channels + in_c];
                    
                    const int row1_base = (base_in_h + 1) * input_w + base_in_w;
                    acc += static_cast<int32_t>(input[row1_base * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 1) * 3 + 0) * in_channels + in_c];
                    acc += static_cast<int32_t>(input[(row1_base + 1) * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 1) * 3 + 1) * in_channels + in_c];
                    acc += static_cast<int32_t>(input[(row1_base + 2) * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 1) * 3 + 2) * in_channels + in_c];
                    
                    const int row2_base = (base_in_h + 2) * input_w + base_in_w;
                    acc += static_cast<int32_t>(input[row2_base * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 2) * 3 + 0) * in_channels + in_c];
                    acc += static_cast<int32_t>(input[(row2_base + 1) * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 2) * 3 + 1) * in_channels + in_c];
                    acc += static_cast<int32_t>(input[(row2_base + 2) * in_channels + in_c] * scale) 
                         * weights[((out_c * 3 + 2) * 3 + 2) * in_channels + in_c];
                }
                
                int8_t result = static_cast<int8_t>(std::max(-127, std::min(127, acc / static_cast<int32_t>(scale))));
                int output_idx = (out_h * output_w + out_w) * out_channels + out_c;
                output[output_idx] = result;
            }
        }
    }
    
    for (int out_c = 0; out_c < out_channels; ++out_c) {
        for (int out_h = 0; out_h < output_h; ++out_h) {
            if (out_h >= interior_start_h && out_h < interior_end_h) continue;
            
            for (int out_w = 0; out_w < output_w; ++out_w) {
                int32_t acc = biases[out_c];
                
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_h = out_h * stride + kh - 1;
                        int in_w = out_w * stride + kw - 1;
                        
                        if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                            for (int in_c = 0; in_c < in_channels; ++in_c) {
                                int input_idx = (in_h * input_w + in_w) * in_channels + in_c;
                                int weight_idx = ((out_c * kernel_h + kh) * kernel_w + kw) * in_channels + in_c;
                                
                                acc += static_cast<int32_t>(input[input_idx] * scale) * weights[weight_idx];
                            }
                        }
                    }
                }
                
                int8_t result = static_cast<int8_t>(std::max(-127, std::min(127, acc / static_cast<int32_t>(scale))));
                int output_idx = (out_h * output_w + out_w) * out_channels + out_c;
                output[output_idx] = result;
            }
        }
        
        for (int out_w = 0; out_w < output_w; ++out_w) {
            if (out_w >= interior_start_w && out_w < interior_end_w) continue;
            
            for (int out_h = interior_start_h; out_h < interior_end_h; ++out_h) {
                int32_t acc = biases[out_c];
                
                // Original code with bounds checking for border pixels
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_h = out_h * stride + kh - 1;  // padding=1
                        int in_w = out_w * stride + kw - 1;
                        
                        if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                            for (int in_c = 0; in_c < in_channels; ++in_c) {
                                int input_idx = (in_h * input_w + in_w) * in_channels + in_c;
                                int weight_idx = ((out_c * kernel_h + kh) * kernel_w + kw) * in_channels + in_c;
                                
                                acc += static_cast<int32_t>(input[input_idx] * scale) * weights[weight_idx];
                            }
                        }
                    }
                }
                
                int8_t result = static_cast<int8_t>(std::max(-127, std::min(127, acc / static_cast<int32_t>(scale))));
                int output_idx = (out_h * output_w + out_w) * out_channels + out_c;
                output[output_idx] = result;
            }
        }
    }
}

FeatureTransformer::FeatureTransformer() : scale(64.0f), num_features(0), output_size(0) {}

bool FeatureTransformer::load_from_stream(std::ifstream& file) {
    // Read scale
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    
    // Read dimensions
    file.read(reinterpret_cast<char*>(&num_features), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&output_size), sizeof(uint32_t));
    
    // Read weights
    long long weight_count = static_cast<long long>(num_features) * output_size;
    if (num_features <= 0 || output_size <= 0 || weight_count <= 0) {
        std::cerr << "Invalid FT dimensions" << std::endl;
        return false;
    }
    weights.resize(static_cast<size_t>(weight_count));
    file.read(reinterpret_cast<char*>(weights.data()), static_cast<std::streamsize>(weight_count * sizeof(int16_t)));
    if (!file.good()) {
        std::cerr << "Failed to read FT weights" << std::endl;
        return false;
    }
    
    // Read bias dimensions and data
    uint32_t bias_count;
    file.read(reinterpret_cast<char*>(&bias_count), sizeof(uint32_t));
    if (bias_count != static_cast<uint32_t>(output_size)) {
        std::cerr << "FT bias count mismatch" << std::endl;
        return false;
    }
    
    biases.resize(bias_count);
    file.read(reinterpret_cast<char*>(biases.data()), bias_count * sizeof(int32_t));
    if (!file.good()) {
        std::cerr << "Failed to read FT biases" << std::endl;
        return false;
    }
    
    return true;
}

void FeatureTransformer::forward(const std::vector<int>& active_features, int16_t* output) const {
    // Initialize with bias
    for (int i = 0; i < output_size; ++i) {
        output[i] = static_cast<int16_t>(biases[i]);
    }
    
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::ft_forward_avx2(active_features, weights.data(), biases.data(), 
                                output, num_features, output_size, scale);
            return;
        }
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::ft_forward_neon(active_features, weights.data(), biases.data(),
                                output, num_features, output_size, scale);
            return;
        }
    #endif
    
    simd::ft_forward_scalar(active_features, weights.data(), biases.data(),
                          output, num_features, output_size, scale);
}

// Chess engine-style incremental updates
void FeatureTransformer::add_feature(int feature_idx, int16_t* accumulator) const {
    if (feature_idx < 0 || feature_idx >= num_features) return;
    
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::add_feature_avx2(feature_idx, weights.data(), accumulator, output_size);
            return;
        }
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::add_feature_neon(feature_idx, weights.data(), accumulator, output_size);
            return;
        }
    #endif
    
    simd::add_feature_scalar(feature_idx, weights.data(), accumulator, output_size);
}

void FeatureTransformer::remove_feature(int feature_idx, int16_t* accumulator) const {
    if (feature_idx < 0 || feature_idx >= num_features) return;
    
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::remove_feature_avx2(feature_idx, weights.data(), accumulator, output_size);
            return;
        }
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::remove_feature_neon(feature_idx, weights.data(), accumulator, output_size);
            return;
        }
    #endif
    
    simd::remove_feature_scalar(feature_idx, weights.data(), accumulator, output_size);
}

void FeatureTransformer::move_feature(int from_idx, int to_idx, int16_t* accumulator) const {
    remove_feature(from_idx, accumulator);
    add_feature(to_idx, accumulator);
}

void FeatureTransformer::update_accumulator(const std::vector<int>& added_features,
                                           const std::vector<int>& removed_features,
                                           int16_t* accumulator) const {
    for (int feature_idx : removed_features) {
        remove_feature(feature_idx, accumulator);
    }
    
    for (int feature_idx : added_features) {
        add_feature(feature_idx, accumulator);
    }
}

void FeatureTransformer::add_feature_simd(int feature_idx, int16_t* accumulator) const {
    add_feature(feature_idx, accumulator);
}

void FeatureTransformer::remove_feature_simd(int feature_idx, int16_t* accumulator) const {
    remove_feature(feature_idx, accumulator);
}

void FeatureTransformer::forward_simd(const std::vector<int>& active_features, int16_t* output) const {
    forward(active_features, output);
}

LayerStack::LayerStack() : l1_size(0), l2_size(0), l3_size(0), l1_scale(64.0f), l1_fact_scale(64.0f), l2_scale(64.0f), output_scale(16.0f) {}

bool LayerStack::load_from_stream(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&l1_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&l2_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&output_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&l1_fact_scale), sizeof(float));
    
    uint32_t l1_out_size, l1_in_size;
    file.read(reinterpret_cast<char*>(&l1_out_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l1_in_size), sizeof(uint32_t));
    if (!file.good() || l1_out_size < 1 || l1_in_size < 1) {
        std::cerr << "Invalid L1 sizes" << std::endl;
        return false;
    }
    
    l1_size = static_cast<int>(l1_in_size);
    l2_size = static_cast<int>(l1_out_size) - 1;
    if (l2_size < 1) {
        std::cerr << "Invalid L2 size computed from L1: " << l2_size << std::endl;
        return false;
    }
    
    {
        long long count = static_cast<long long>(l1_out_size) * l1_in_size;
        l1_weights.resize(static_cast<size_t>(count));
        file.read(reinterpret_cast<char*>(l1_weights.data()), static_cast<std::streamsize>(count));
        if (!file.good()) { std::cerr << "Failed to read L1 weights" << std::endl; return false; }
    }
    
    uint32_t l1_bias_count;
    file.read(reinterpret_cast<char*>(&l1_bias_count), sizeof(uint32_t));
    l1_biases.resize(l1_bias_count);
    file.read(reinterpret_cast<char*>(l1_biases.data()), l1_bias_count * sizeof(int32_t));
    if (!file.good()) { std::cerr << "Failed to read L1 biases" << std::endl; return false; }
    
    uint32_t l1_fact_out_size, l1_fact_in_size;
    file.read(reinterpret_cast<char*>(&l1_fact_out_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l1_fact_in_size), sizeof(uint32_t));
    
    if (l1_fact_in_size != static_cast<uint32_t>(l1_size)) {
        std::cerr << "Invalid L1 factorization layer input size: " << l1_fact_in_size << " != " << l1_size << std::endl;
        return false;
    }
    if (l1_fact_out_size <= static_cast<uint32_t>(l2_size)) {
        std::cerr << "L1 factorization output too small: " << l1_fact_out_size << " <= L2 size " << l2_size << std::endl;
        return false;
    }
    
    {
        long long count = static_cast<long long>(l1_fact_out_size) * l1_fact_in_size;
        l1_fact_weights.resize(static_cast<size_t>(count));
        file.read(reinterpret_cast<char*>(l1_fact_weights.data()), static_cast<std::streamsize>(count));
        if (!file.good()) { std::cerr << "Failed to read L1 factorization weights" << std::endl; return false; }
    }
    
    uint32_t l1_fact_bias_count;
    file.read(reinterpret_cast<char*>(&l1_fact_bias_count), sizeof(uint32_t));
    l1_fact_biases.resize(l1_fact_bias_count);
    file.read(reinterpret_cast<char*>(l1_fact_biases.data()), l1_fact_bias_count * sizeof(int32_t));
    if (!file.good()) { std::cerr << "Failed to read L1 factorization biases" << std::endl; return false; }
    
    uint32_t l2_out_size, l2_in_size;
    file.read(reinterpret_cast<char*>(&l2_out_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l2_in_size), sizeof(uint32_t));
    
    if (l2_in_size != static_cast<uint32_t>(l2_size * 2)) {
        std::cerr << "Invalid L2 layer dimensions: " << l2_in_size << " -> " << l2_out_size << std::endl;
        return false;
    }
    
    l3_size = l2_out_size;
    
    {
        long long count = static_cast<long long>(l2_out_size) * l2_in_size;
        l2_weights.resize(static_cast<size_t>(count));
        file.read(reinterpret_cast<char*>(l2_weights.data()), static_cast<std::streamsize>(count));
        if (!file.good()) { std::cerr << "Failed to read L2 weights" << std::endl; return false; }
    }
    
    uint32_t l2_bias_count;
    file.read(reinterpret_cast<char*>(&l2_bias_count), sizeof(uint32_t));
    l2_biases.resize(l2_bias_count);
    file.read(reinterpret_cast<char*>(l2_biases.data()), l2_bias_count * sizeof(int32_t));
    if (!file.good()) { std::cerr << "Failed to read L2 biases" << std::endl; return false; }
    
    // Read output layer
    uint32_t out_out_size, out_in_size;
    file.read(reinterpret_cast<char*>(&out_out_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&out_in_size), sizeof(uint32_t));
    
    if (out_in_size != static_cast<uint32_t>(l3_size) || out_out_size < 1) {
        std::cerr << "Invalid output layer dimensions: " << out_in_size << " -> " << out_out_size << std::endl;
        return false;
    }
    out_classes = static_cast<int>(out_out_size);
    
    {
    long long count = static_cast<long long>(out_out_size) * out_in_size;
        output_weights.resize(static_cast<size_t>(count));
        file.read(reinterpret_cast<char*>(output_weights.data()), static_cast<std::streamsize>(count));
        if (!file.good()) { std::cerr << "Failed to read output weights" << std::endl; return false; }
    }
    
    uint32_t out_bias_count;
    file.read(reinterpret_cast<char*>(&out_bias_count), sizeof(uint32_t));
    output_biases.resize(out_bias_count);
    file.read(reinterpret_cast<char*>(output_biases.data()), out_bias_count * sizeof(int32_t));
    if (!file.good()) { std::cerr << "Failed to read output biases" << std::endl; return false; }
    
    return true;
}

float LayerStack::forward(const int16_t* input, int layer_stack_index) const {
    // Suppress unused parameter warning (layer_stack_index may be used in future multi-bucket scenarios)
    (void)layer_stack_index;
    

    
    // L1 combined layer: l1_size -> (l2_size + 1) with ClippedReLU
    if (l2_size < 1 || l1_size < 1) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::vector<int8_t> l1_combined_output(static_cast<size_t>(l2_size + 1));
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::dense_forward_avx2(input, l1_weights.data(), l1_biases.data(), 
                                   l1_combined_output.data(), l1_size, l2_size + 1, l1_scale);
        } else
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::dense_forward_neon(input, l1_weights.data(), l1_biases.data(),
                                   l1_combined_output.data(), l1_size, l2_size + 1, l1_scale);
        } else
    #endif
        {
            simd::dense_forward_scalar(input, l1_weights.data(), l1_biases.data(),
                                     l1_combined_output.data(), l1_size, l2_size + 1, l1_scale);
        }
    
    float l1c_out = static_cast<float>(l1_combined_output[static_cast<size_t>(l2_size)]) / l1_scale;
    
    std::vector<int8_t> l1_fact_output(l1_fact_biases.size());
    simd::dense_forward_scalar(input, l1_fact_weights.data(), l1_fact_biases.data(),
                             l1_fact_output.data(), l1_size, l1_fact_biases.size(), l1_fact_scale);
    if (l1_fact_output.size() <= static_cast<size_t>(l2_size)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float l1f_out = static_cast<float>(l1_fact_output[static_cast<size_t>(l2_size)]) / l1_fact_scale;
    
    std::vector<int16_t> l1_expanded(static_cast<size_t>(l2_size) * 2U);
    
    int i = 0;
    for (; i <= l2_size - 4; i += 4) {
        int32_t squared0 = static_cast<int32_t>(l1_combined_output[i]) * static_cast<int32_t>(l1_combined_output[i]);
        squared0 = (squared0 * 127) / 128;
        l1_expanded[i] = static_cast<int16_t>(std::max(0, std::min(127, squared0)));
        l1_expanded[i + l2_size] = static_cast<int16_t>(l1_combined_output[i]);
        
        int32_t squared1 = static_cast<int32_t>(l1_combined_output[i+1]) * static_cast<int32_t>(l1_combined_output[i+1]);
        squared1 = (squared1 * 127) / 128;
        l1_expanded[i+1] = static_cast<int16_t>(std::max(0, std::min(127, squared1)));
        l1_expanded[i+1 + l2_size] = static_cast<int16_t>(l1_combined_output[i+1]);
        
        int32_t squared2 = static_cast<int32_t>(l1_combined_output[i+2]) * static_cast<int32_t>(l1_combined_output[i+2]);
        squared2 = (squared2 * 127) / 128;
        l1_expanded[i+2] = static_cast<int16_t>(std::max(0, std::min(127, squared2)));
        l1_expanded[i+2 + l2_size] = static_cast<int16_t>(l1_combined_output[i+2]);
        
        int32_t squared3 = static_cast<int32_t>(l1_combined_output[i+3]) * static_cast<int32_t>(l1_combined_output[i+3]);
        squared3 = (squared3 * 127) / 128;
        l1_expanded[i+3] = static_cast<int16_t>(std::max(0, std::min(127, squared3)));
        l1_expanded[i+3 + l2_size] = static_cast<int16_t>(l1_combined_output[i+3]);
    }
    
    for (; i < l2_size; ++i) {
        int32_t squared = static_cast<int32_t>(l1_combined_output[i]) * static_cast<int32_t>(l1_combined_output[i]);
        squared = (squared * 127) / 128;
        l1_expanded[i] = static_cast<int16_t>(std::max(0, std::min(127, squared)));
        l1_expanded[i + l2_size] = static_cast<int16_t>(l1_combined_output[i]);
    }
    
    std::vector<int8_t> l2_output(l3_size);
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::dense_forward_avx2(l1_expanded.data(), l2_weights.data(), l2_biases.data(),
                                   l2_output.data(), l2_size * 2, l3_size, l2_scale);
        } else
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::dense_forward_neon(l1_expanded.data(), l2_weights.data(), l2_biases.data(),
                                   l2_output.data(), l2_size * 2, l3_size, l2_scale);
        } else
    #endif
        {
            simd::dense_forward_scalar(l1_expanded.data(), l2_weights.data(), l2_biases.data(),
                                     l2_output.data(), l2_size * 2, l3_size, l2_scale);
        }
    
    int32_t output_acc = output_biases[0];
    for (int j = 0; j < l3_size; ++j) {
        output_acc += static_cast<int32_t>(l2_output[j]) * static_cast<int32_t>(output_weights[j]);
    }
    
    float l3c = static_cast<float>(output_acc) / output_scale;
    
    return l3c + l1f_out + l1c_out;
}

std::vector<float> LayerStack::forward_multiclass(const int16_t* input, int layer_stack_index) const {
    (void)layer_stack_index;
    std::vector<float> logits(static_cast<size_t>(std::max(1, out_classes)), 0.0f);

    // L1 combined
    if (l2_size < 1 || l1_size < 1) {
        return logits;
    }
    std::vector<int8_t> l1_combined_output(static_cast<size_t>(l2_size + 1));
    simd::dense_forward_scalar(input, l1_weights.data(), l1_biases.data(),
                               l1_combined_output.data(), l1_size, l2_size + 1, l1_scale);
    float l1c_out = static_cast<float>(l1_combined_output[static_cast<size_t>(l2_size)]) / l1_scale;

    std::vector<int8_t> l1_fact_output(l1_fact_biases.size());
    simd::dense_forward_scalar(input, l1_fact_weights.data(), l1_fact_biases.data(),
                               l1_fact_output.data(), l1_size, l1_fact_biases.size(), l1_fact_scale);
    if (l1_fact_output.size() <= static_cast<size_t>(l2_size)) {
        return logits;
    }
    float l1f_out = static_cast<float>(l1_fact_output[static_cast<size_t>(l2_size)]) / l1_fact_scale;

    std::vector<int16_t> l1_expanded(static_cast<size_t>(l2_size) * 2U);
    for (int i = 0; i < l2_size; ++i) {
        int32_t squared = static_cast<int32_t>(l1_combined_output[i]) * static_cast<int32_t>(l1_combined_output[i]);
        squared = (squared * 127) / 128;
        l1_expanded[i] = static_cast<int16_t>(std::max(0, std::min(127, squared)));
        l1_expanded[i + l2_size] = static_cast<int16_t>(l1_combined_output[i]);
    }

    std::vector<int8_t> l2_output(l3_size);
    simd::dense_forward_scalar(l1_expanded.data(), l2_weights.data(), l2_biases.data(),
                               l2_output.data(), l2_size * 2, l3_size, l2_scale);

    for (int cls = 0; cls < out_classes; ++cls) {
        int32_t acc = output_biases[cls];
        for (int j = 0; j < l3_size; ++j) {
            int idx = cls * l3_size + j;
            acc += static_cast<int32_t>(l2_output[j]) * static_cast<int32_t>(output_weights[idx]);
        }
        float l3c = static_cast<float>(acc) / output_scale;
        logits[static_cast<size_t>(cls)] = l3c + l1f_out + l1c_out;
    }

    return logits;
}

NNUEEvaluator::NNUEEvaluator() : num_features_(0), l1_size_(0), l2_size_(0), l3_size_(0), 
                                num_ls_buckets_(0), grid_size_(0), num_channels_per_square_(0),
                                visual_threshold_(0.0f), nnue2score_(600.0f), quantized_one_(127.0f),
                                accumulator_dirty_(true), incremental_enabled_(false) {
}

bool NNUEEvaluator::load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        return false;
    }
    
    // Read and verify magic number
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "NNUE", 4) != 0) {
        std::cerr << "Invalid magic number" << std::endl;
        return false;
    }
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != 2) {
        std::cerr << "Unsupported version: " << version << std::endl;
        return false;
    }
    
    // Read architecture metadata
    uint32_t num_features, l1_size, l2_size, l3_size;
    file.read(reinterpret_cast<char*>(&num_features), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l1_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l2_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l3_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_ls_buckets_), sizeof(uint32_t));
    
    // Store architecture parameters
    num_features_ = num_features;
    l1_size_ = l1_size;
    l2_size_ = l2_size;
    l3_size_ = l3_size;
    
    // Read quantization parameters
    file.read(reinterpret_cast<char*>(&nnue2score_), sizeof(float));
    file.read(reinterpret_cast<char*>(&quantized_one_), sizeof(float));
    file.read(reinterpret_cast<char*>(&visual_threshold_), sizeof(float));
    
    // Load conv layer
    if (!conv_layer_.load_from_stream(file)) {
        std::cerr << "Failed to load conv layer" << std::endl;
        return false;
    }
    
    // Calculate grid parameters from conv layer and num_features
    num_channels_per_square_ = conv_layer_.out_channels;
    if (num_channels_per_square_ <= 0 || num_features == 0 || num_features % static_cast<uint32_t>(num_channels_per_square_) != 0) {
        std::cerr << "Invalid feature/channel configuration" << std::endl;
        return false;
    }
    grid_size_ = static_cast<int>(std::sqrt(num_features / num_channels_per_square_));
    
    if (static_cast<uint32_t>(grid_size_ * grid_size_ * num_channels_per_square_) != num_features) {
        std::cerr << "Invalid feature grid calculation" << std::endl;
        return false;
    }
    
    // Load feature transformer
    if (!feature_transformer_.load_from_stream(file)) {
        std::cerr << "Failed to load feature transformer" << std::endl;
        return false;
    }
    
    // Verify feature transformer architecture matches
    if (feature_transformer_.num_features != num_features_ || 
        feature_transformer_.output_size != l1_size_) {
        std::cerr << "Feature transformer architecture mismatch" << std::endl;
        return false;
    }
    
    // Load layer stacks
    layer_stacks_.clear();
    layer_stacks_.reserve(num_ls_buckets_);
    for (int i = 0; i < num_ls_buckets_; ++i) {
        layer_stacks_.emplace_back();
        if (!layer_stacks_[i].load_from_stream(file)) {
            std::cerr << "Failed to load layer stack " << i << std::endl;
            return false;
        }
        
        // Verify layer stack architecture
        if (layer_stacks_[i].l1_size != l1_size_ ||
            layer_stacks_[i].l2_size != l2_size_ ||
            layer_stacks_[i].l3_size != l3_size_) {
            std::cerr << "Layer stack architecture mismatch" << std::endl;
            return false;
        }
    }
    
    // Initialize working buffers with correct sizes
    long long conv_output_size = static_cast<long long>(grid_size_) * grid_size_ * num_channels_per_square_;
    if (conv_output_size <= 0) {
        std::cerr << "Invalid conv output size" << std::endl;
        return false;
    }
    conv_output_.resize(static_cast<size_t>(conv_output_size));
    
    feature_grid_ = std::make_unique<DynamicGrid>(grid_size_, num_channels_per_square_);
    
    if (l1_size_ <= 0) { std::cerr << "Invalid l1_size" << std::endl; return false; }
    ft_output_.resize(static_cast<size_t>(l1_size_));
    
    // Initialize chess engine-style accumulator management
    accumulator_.resize(l1_size_);
    backup_accumulator_.resize(l1_size_);
    accumulator_dirty_ = true;
    incremental_enabled_ = true;  // Enable by default
    
    return true;
}

float NNUEEvaluator::evaluate(const float* image_data, int image_h, int image_w, int layer_stack_index) const {
    if (layer_stack_index >= num_ls_buckets_) {
        layer_stack_index = 0;
    }
    
    // Calculate conv stride to not exceed target grid size (avoid buffer overruns)
    if (grid_size_ <= 0) return std::numeric_limits<float>::quiet_NaN();
    int conv_stride;
    if (grid_size_ > 1) {
        // ceil((image_h - 1) / (grid_size_ - 1)) to ensure output_h <= grid_size_
        int num = image_h - 1;
        int den = grid_size_ - 1;
        conv_stride = (num + den - 1) / den;
    } else {
        conv_stride = std::max(1, image_h);  // collapse to 1 cell
    }
    if (conv_stride < 1) conv_stride = 1;
    
    // Step 1: Convolution image_h x image_w x 3 -> up to grid_size x grid_size x num_channels_per_square
    // Zero the destination buffer so any cells beyond produced output remain zero
    std::fill(conv_output_.data(), conv_output_.data() + conv_output_.size(), static_cast<int8_t>(0));
    conv_layer_.forward(image_data, conv_output_.data(), image_h, image_w, conv_stride);
    
    // Step 2: Convert to feature grid and extract active features efficiently
    feature_grid_->from_conv_output(conv_output_.data(), visual_threshold_);
    feature_grid_->extract_features(active_features_);
    
    // Step 3: Feature transformer (sparse -> dense)
    feature_transformer_.forward(active_features_, ft_output_.data());
    
    // Step 4: Apply clipped ReLU to feature transformer output
    for (int i = 0; i < l1_size_; ++i) {
        ft_output_[i] = std::max(static_cast<int16_t>(0), 
                                std::min(ft_output_[i], static_cast<int16_t>(quantized_one_)));
    }
    
    // Step 5: Pass through layer stack
    if (layer_stack_index < 0 || layer_stack_index >= static_cast<int>(layer_stacks_.size())) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float raw_output = layer_stacks_[layer_stack_index].forward(ft_output_.data(), layer_stack_index);
    
    return raw_output;
}

std::vector<float> NNUEEvaluator::evaluate_logits(const float* image_data, int image_h, int image_w, int layer_stack_index) const {
    if (layer_stack_index >= num_ls_buckets_) {
        layer_stack_index = 0;
    }
    if (grid_size_ <= 0) return {};

    int conv_stride;
    if (grid_size_ > 1) {
        int num = image_h - 1;
        int den = grid_size_ - 1;
        conv_stride = (num + den - 1) / den;
    } else {
        conv_stride = std::max(1, image_h);
    }
    if (conv_stride < 1) conv_stride = 1;

    std::fill(conv_output_.data(), conv_output_.data() + conv_output_.size(), static_cast<int8_t>(0));
    conv_layer_.forward(image_data, conv_output_.data(), image_h, image_w, conv_stride);

    feature_grid_->from_conv_output(conv_output_.data(), visual_threshold_);
    feature_grid_->extract_features(active_features_);
    feature_transformer_.forward(active_features_, ft_output_.data());
    for (int i = 0; i < l1_size_; ++i) {
        ft_output_[i] = std::max(static_cast<int16_t>(0),
                                 std::min(ft_output_[i], static_cast<int16_t>(quantized_one_)));
    }
    if (layer_stack_index < 0 || layer_stack_index >= static_cast<int>(layer_stacks_.size())) {
        return {};
    }
    return layer_stacks_[layer_stack_index].forward_multiclass(ft_output_.data(), layer_stack_index);
}

// extract_features is now implemented inline in the header

// Chess engine-style incremental evaluation
float NNUEEvaluator::evaluate_incremental(const std::vector<int>& current_features, int layer_stack_index) const {
    if (layer_stack_index >= num_ls_buckets_) {
        layer_stack_index = 0;
    }
    
    if (!incremental_enabled_ || accumulator_dirty_) {
        // Full refresh needed
        refresh_accumulator(current_features);
        last_active_features_ = current_features;
        accumulator_dirty_ = false;
    } else {
        // Incremental update - compute difference from last state
        std::vector<int> added_features, removed_features;
        
        // Find features to remove (in last but not in current)
        for (int feature : last_active_features_) {
            if (std::find(current_features.begin(), current_features.end(), feature) == current_features.end()) {
                removed_features.push_back(feature);
            }
        }
        
        // Find features to add (in current but not in last)
        for (int feature : current_features) {
            if (std::find(last_active_features_.begin(), last_active_features_.end(), feature) == last_active_features_.end()) {
                added_features.push_back(feature);
            }
        }
        
        // Apply incremental update
        if (!added_features.empty() || !removed_features.empty()) {
            update_features(added_features, removed_features);
            last_active_features_ = current_features;
        }
    }
    
    // Copy accumulator to ft_output for processing
    for (int i = 0; i < l1_size_; ++i) {
        ft_output_[i] = accumulator_[i];
    }
    
    // Apply clipped ReLU
    for (int i = 0; i < l1_size_; ++i) {
        ft_output_[i] = std::max(static_cast<int16_t>(0), 
                                std::min(ft_output_[i], static_cast<int16_t>(quantized_one_)));
    }
    
    // Pass through layer stack
    float raw_output = layer_stacks_[layer_stack_index].forward(ft_output_.data(), layer_stack_index);
    
    return raw_output;
}

// Chess engine-style accumulator management
void NNUEEvaluator::save_accumulator() const {
    // Backup current accumulator state
    for (int i = 0; i < l1_size_; ++i) {
        backup_accumulator_[i] = accumulator_[i];
    }
}

void NNUEEvaluator::restore_accumulator() const {
    // Restore from backup
    for (int i = 0; i < l1_size_; ++i) {
        accumulator_[i] = backup_accumulator_[i];
    }
}

void NNUEEvaluator::refresh_accumulator(const std::vector<int>& features) const {
    // Initialize with bias (like chess engine full reset)
    for (int i = 0; i < l1_size_; ++i) {
        accumulator_[i] = static_cast<int16_t>(feature_transformer_.biases[i]);
    }
    
    // Add all active features using SIMD-optimized operations
    for (int feature_idx : features) {
        feature_transformer_.add_feature(feature_idx, accumulator_.data());
    }
}

void NNUEEvaluator::update_features(const std::vector<int>& added, const std::vector<int>& removed) const {
    // Use optimized batch update (like chess engine incremental updates)
    feature_transformer_.update_accumulator(added, removed, accumulator_.data());
}



LinearDepthwiseBlock::LinearDepthwiseBlock()
     : pw_expand_scale(64.0f), dw_conv_scale(64.0f), pw_project_scale(64.0f),
      in_channels(0), mid_channels(0), out_channels(0), stride(1) {}

bool LinearDepthwiseBlock::load_from_stream(std::ifstream& file) {
    // Read scales (matches new architecture: pw_expand, dw_conv, pw_project)
    file.read(reinterpret_cast<char*>(&pw_expand_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&dw_conv_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&pw_project_scale), sizeof(float));
    
    // Read dimensions
    file.read(reinterpret_cast<char*>(&in_channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&mid_channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&out_channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&stride), sizeof(uint32_t));
    
    // Read pointwise expansion weights (1x1 conv: in_channels -> mid_channels)
    int pw_expand_weight_count = mid_channels * in_channels;
    pw_expand_weights.resize(pw_expand_weight_count);
    file.read(reinterpret_cast<char*>(pw_expand_weights.data()), pw_expand_weight_count);
    
    // Read pointwise expansion biases  
    uint32_t pw_expand_bias_count;
    file.read(reinterpret_cast<char*>(&pw_expand_bias_count), sizeof(uint32_t));
    if (pw_expand_bias_count != static_cast<uint32_t>(mid_channels)) {
        std::cerr << "Linear block pw_expand bias count mismatch" << std::endl;
        return false;
    }
    pw_expand_biases.resize(pw_expand_bias_count);
    file.read(reinterpret_cast<char*>(pw_expand_biases.data()), pw_expand_bias_count * sizeof(int32_t));
    
    // Read depthwise conv weights (3x3, groups=mid_channels)
    int dw_conv_weight_count = mid_channels * 9;  // 3x3 kernel per channel
    dw_conv_weights.resize(dw_conv_weight_count);
    file.read(reinterpret_cast<char*>(dw_conv_weights.data()), dw_conv_weight_count);
    
    // Read pointwise projection weights (1x1 conv: mid_channels -> out_channels)
    int pw_project_weight_count = out_channels * mid_channels;
    pw_project_weights.resize(pw_project_weight_count);
    file.read(reinterpret_cast<char*>(pw_project_weights.data()), pw_project_weight_count);
    
    // Read bias count and data
    uint32_t bias_count;
    file.read(reinterpret_cast<char*>(&bias_count), sizeof(uint32_t));
    if (bias_count != static_cast<uint32_t>(out_channels)) {
        std::cerr << "Linear block bias count mismatch" << std::endl;
        return false;
    }
    // Skip bias data (not used in forward pass)
    file.seekg(bias_count * sizeof(int32_t), std::ios::cur);
    
    return file.good();
}

void LinearDepthwiseBlock::forward(const int8_t* input, int8_t* output, int input_h, int input_w, const MemoryPool* pool) const {
    // Calculate output dimensions after depthwise conv (the only layer that can change spatial size)
    int out_h = (input_h - 3 + 2) / stride + 1;  // After depthwise conv with padding=1
    int out_w = (input_w - 3 + 2) / stride + 1;
    
    // Use memory pool for temporary buffers if provided
    size_t pw_expand_size = input_h * input_w * mid_channels;  // After pointwise expansion
    size_t dw_conv_size = out_h * out_w * mid_channels;       // After depthwise conv
    
    std::vector<int8_t> local_pw_expand, local_dw_conv;
    int8_t* pw_expand_output;
    int8_t* dw_conv_output;
    
    std::unique_ptr<MemoryPool::BufferLock> pool_buffer1, pool_buffer2;
    if (pool) {
        pool_buffer1 = std::make_unique<MemoryPool::BufferLock>(pool->get_buffer_lock(pw_expand_size));
        pool_buffer2 = std::make_unique<MemoryPool::BufferLock>(pool->get_buffer_lock(dw_conv_size));
        pw_expand_output = pool_buffer1->get();
        dw_conv_output = pool_buffer2->get();
    } else {
        local_pw_expand.resize(pw_expand_size);
        local_dw_conv.resize(dw_conv_size);
        pw_expand_output = local_pw_expand.data();
        dw_conv_output = local_dw_conv.data();
    }
    
    // 1. Pointwise expansion (1x1 conv): in_channels -> mid_channels with ReLU6
    for (int out_c = 0; out_c < mid_channels; ++out_c) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                int32_t acc = pw_expand_biases[out_c];
                
                for (int in_c = 0; in_c < in_channels; ++in_c) {
                    int input_idx = (h * input_w + w) * in_channels + in_c;
                    int weight_idx = out_c * in_channels + in_c;
                    acc += static_cast<int32_t>(input[input_idx]) * static_cast<int32_t>(pw_expand_weights[weight_idx]);
                }
                
                int result = acc / static_cast<int32_t>(pw_expand_scale);
                result = std::max(0, std::min(6, result));  // ReLU6 activation
                
                int output_idx = (h * input_w + w) * mid_channels + out_c;
                pw_expand_output[output_idx] = static_cast<int8_t>(std::max(-127, std::min(127, result)));
            }
        }
    }
    
    // 2. Depthwise conv (3x3, groups=mid_channels): mid_channels -> mid_channels with ReLU6
    for (int c = 0; c < mid_channels; ++c) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                int32_t acc = 0;
                
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int in_y = y * stride + kh - 1;  // padding=1
                        int in_x = x * stride + kw - 1;
                        
                        if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                            int input_idx = (in_y * input_w + in_x) * mid_channels + c;
                            int weight_idx = (c * 3 + kh) * 3 + kw;
                            acc += static_cast<int32_t>(pw_expand_output[input_idx]) * static_cast<int32_t>(dw_conv_weights[weight_idx]);
                        }
                    }
                }
                
                int result = acc / static_cast<int32_t>(dw_conv_scale);
                result = std::max(0, std::min(6, result));  // ReLU6 activation
                
                int output_idx = (y * out_w + x) * mid_channels + c;
                dw_conv_output[output_idx] = static_cast<int8_t>(std::max(-127, std::min(127, result)));
            }
        }
    }
    
    // 3. Pointwise projection (1x1 conv): mid_channels -> out_channels (no activation)
    for (int out_c = 0; out_c < out_channels; ++out_c) {
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                int32_t acc = 0;  // No bias in PyTorch pw_project
                
                for (int in_c = 0; in_c < mid_channels; ++in_c) {
                    int input_idx = (h * out_w + w) * mid_channels + in_c;
                    int weight_idx = out_c * mid_channels + in_c;
                    acc += static_cast<int32_t>(dw_conv_output[input_idx]) * static_cast<int32_t>(pw_project_weights[weight_idx]);
                }
                
                int result = acc / static_cast<int32_t>(pw_project_scale);
                // No activation for projection layer
                
                int output_idx = (h * out_w + w) * out_channels + out_c;
                output[output_idx] = static_cast<int8_t>(std::max(-127, std::min(127, result)));
            }
        }
    }
}

// DenseLinearDepthwiseBlock implementation
DenseLinearDepthwiseBlock::DenseLinearDepthwiseBlock() : use_skip_connection(false) {}

bool DenseLinearDepthwiseBlock::load_from_stream(std::ifstream& file) {
    // Load the inner linear block and skip connection flag
    return linear_block.load_from_stream(file);
}

void DenseLinearDepthwiseBlock::forward(const int8_t* input, int8_t* output, int input_h, int input_w, const MemoryPool* pool) const {
    // Run the linear block with memory pool
    linear_block.forward(input, output, input_h, input_w, pool);
    
    // Add skip connection if enabled and dimensions match
    if (use_skip_connection && linear_block.in_channels == linear_block.out_channels && linear_block.stride == 1) {
        int size = input_h * input_w * linear_block.out_channels;
        for (int i = 0; i < size; ++i) {
            int result = static_cast<int32_t>(output[i]) + static_cast<int32_t>(input[i]);
            output[i] = static_cast<int8_t>(std::max(-127, std::min(127, result)));
        }
    }
}

// LinearLayer implementation
LinearLayer::LinearLayer() : scale(64.0f), in_features(0), out_features(0) {}

bool LinearLayer::load_from_stream(std::ifstream& file) {
    // Read scale
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    
    // Read dimensions
    file.read(reinterpret_cast<char*>(&in_features), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&out_features), sizeof(uint32_t));
    
    // Read weights
    int weight_count = out_features * in_features;
    weights.resize(weight_count);
    file.read(reinterpret_cast<char*>(weights.data()), weight_count);
    
    // Read bias dimensions and data
    uint32_t bias_count;
    file.read(reinterpret_cast<char*>(&bias_count), sizeof(uint32_t));
    if (bias_count != static_cast<uint32_t>(out_features)) {
        std::cerr << "Linear layer bias count mismatch" << std::endl;
        return false;
    }
    
    biases.resize(bias_count);
    file.read(reinterpret_cast<char*>(biases.data()), bias_count * sizeof(int32_t));
    
    return file.good();
}

void LinearLayer::forward(const int8_t* input, float* output) const {
    for (int out_f = 0; out_f < out_features; ++out_f) {
        int32_t acc = biases[out_f];
        
        for (int in_f = 0; in_f < in_features; ++in_f) {
            int weight_idx = out_f * in_features + in_f;
            acc += static_cast<int32_t>(input[in_f]) * static_cast<int32_t>(weights[weight_idx]);
        }
        
        // Convert to float with proper scaling
        output[out_f] = static_cast<float>(acc) / scale;
    }
}

// EtinyNetEvaluator implementation
EtinyNetEvaluator::EtinyNetEvaluator() 
    : variant_("1.0"), num_classes_(1000), input_size_(112), conv_channels_(32), final_channels_(512),
      use_asq_(false), asq_bits_(4), lambda_param_(2.0f) {}

bool EtinyNetEvaluator::load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open EtinyNet file: " << path << std::endl;
        return false;
    }
    
    // Load header
    if (!load_header(file)) {
        std::cerr << "Failed to load EtinyNet header" << std::endl;
        return false;
    }
    
    // Load layers
    if (!load_layers(file)) {
        std::cerr << "Failed to load EtinyNet layers" << std::endl;
        return false;
    }
    
    // Calculate layer dimensions and allocate buffers
    calculate_layer_dimensions();
    allocate_working_buffers();
    
    // Debug output removed for clean inference
    
    return true;
}

bool EtinyNetEvaluator::load_header(std::ifstream& file) {
    // Read and verify magic number
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "ETNY", 4) != 0) {
        std::cerr << "Invalid EtinyNet magic number" << std::endl;
        return false;
    }
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != 1) {
        std::cerr << "Unsupported EtinyNet version: " << version << std::endl;
        return false;
    }
    
    // Read variant string
    uint32_t variant_len;
    file.read(reinterpret_cast<char*>(&variant_len), sizeof(uint32_t));
    std::vector<char> variant_chars(variant_len);
    file.read(variant_chars.data(), variant_len);
    variant_ = std::string(variant_chars.begin(), variant_chars.end());
    
    // Read architecture metadata
    file.read(reinterpret_cast<char*>(&num_classes_), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&input_size_), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&conv_channels_), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&final_channels_), sizeof(uint32_t));
    
    // Read ASQ parameters
    file.read(reinterpret_cast<char*>(&use_asq_), sizeof(bool));
    if (use_asq_) {
        file.read(reinterpret_cast<char*>(&asq_bits_), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&lambda_param_), sizeof(float));
    }
    
    return file.good();
}

bool EtinyNetEvaluator::load_layers(std::ifstream& file) {
    // Read number of layers in the sequence
    uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
    
    // Clear existing layer storages and sequence
    lb_layers_.clear();
    dlb_layers_.clear();
    classifier_.reset();
    layer_sequence_.clear();
    layer_sequence_.reserve(num_layers);
    
    // Load each layer based on its type
    for (uint32_t i = 0; i < num_layers; ++i) {
        // Read layer type identifier
        uint32_t layer_type;
        file.read(reinterpret_cast<char*>(&layer_type), sizeof(uint32_t));
        
        LayerInfo info;
        info.type = static_cast<EtinyNetLayerType>(layer_type);
        
        switch (layer_type) {
            case 0: { // ConvLayer (STANDARD_CONV)
                auto layer = std::make_unique<ConvLayer>();
                if (!layer->load_from_stream(file)) {
                    std::cerr << "Failed to load ConvLayer layer " << i << std::endl;
                    return false;
                }
                
                info.index = static_cast<int>(conv_layers_.size());
                // Note: output dimensions would be calculated during layer execution
                info.output_h = info.output_w = info.output_c = 0;  // To be calculated
                
                conv_layers_.emplace_back(std::move(layer));
                layer_sequence_.emplace_back(info);
                break;
            }
            
            case 1: { // LinearDepthwiseBlock (LINEAR_DEPTHWISE)
                auto layer = std::make_unique<LinearDepthwiseBlock>();
                if (!layer->load_from_stream(file)) {
                    std::cerr << "Failed to load LinearDepthwiseBlock layer " << i << std::endl;
                    return false;
                }
                
                info.index = static_cast<int>(lb_layers_.size());
                info.output_h = info.output_w = info.output_c = 0;  // To be calculated
                
                lb_layers_.emplace_back(std::move(layer));
                layer_sequence_.emplace_back(info);
                break;
            }
            
            case 2: { // DenseLinearDepthwiseBlock (DENSE_LINEAR_DEPTHWISE)
                auto layer = std::make_unique<DenseLinearDepthwiseBlock>();
                if (!layer->load_from_stream(file)) {
                    std::cerr << "Failed to load DenseLinearDepthwiseBlock layer " << i << std::endl;
                    return false;
                }
                
                info.index = static_cast<int>(dlb_layers_.size());
                info.output_h = info.output_w = info.output_c = 0;  // To be calculated
                
                dlb_layers_.emplace_back(std::move(layer));
                layer_sequence_.emplace_back(info);
                break;
            }
            
            case 3: { // LinearLayer (LINEAR_CLASSIFIER)
                classifier_ = std::make_unique<LinearLayer>();
                if (!classifier_->load_from_stream(file)) {
                    std::cerr << "Failed to load LinearLayer classifier" << std::endl;
                    return false;
                }
                
                info.index = 0;  // Only one classifier
                info.output_h = info.output_w = 1;  // Linear output
                info.output_c = classifier_->out_features;
                
                layer_sequence_.emplace_back(info);
                break;
            }
            
            default:
                std::cerr << "Unknown layer type: " << layer_type << " at layer " << i << std::endl;
                return false;
        }
    }
    
    if (!classifier_) {
        std::cerr << "No classifier layer found in EtinyNet model" << std::endl;
        return false;
    }
    
    // Debug output removed for clean inference
    return true;
}

void EtinyNetEvaluator::calculate_layer_dimensions() {
    // Calculate output dimensions for each layer in the sequence
    int current_h = input_size_;
    int current_w = input_size_;
    int current_c = 3;  // RGB input
    
    for (auto& layer_info : layer_sequence_) {
        switch (layer_info.type) {
            case EtinyNetLayerType::STANDARD_CONV: {
                if (layer_info.index < static_cast<int>(conv_layers_.size())) {
                    const auto& layer = conv_layers_[layer_info.index];
                    
                    // Standard conv: stride=2, padding=1, kernel=3 for initial conv  
                    int stride = 2;  
                    int padding = 1;
                    int kernel = 3;
                    
                    int out_h = (current_h + 2 * padding - kernel) / stride + 1;
                    int out_w = (current_w + 2 * padding - kernel) / stride + 1;
                    int out_c = layer->out_channels;
                    
                    layer_info.output_h = out_h;
                    layer_info.output_w = out_w;
                    layer_info.output_c = out_c;
                    
                    current_h = out_h;
                    current_w = out_w;
                    current_c = out_c;
                }
                break;
            }
            
            case EtinyNetLayerType::LINEAR_DEPTHWISE: {
                if (layer_info.index < static_cast<int>(lb_layers_.size())) {
                    const auto& layer = lb_layers_[layer_info.index];
                    
                    int out_h = (current_h - 3 + 2) / layer->stride + 1;  // Assuming 3x3 kernel, padding=1
                    int out_w = (current_w - 3 + 2) / layer->stride + 1;
                    int out_c = layer->out_channels;
                    
                    layer_info.output_h = out_h;
                    layer_info.output_w = out_w;
                    layer_info.output_c = out_c;
                    
                    current_h = out_h;
                    current_w = out_w;
                    current_c = out_c;
                }
                break;
            }
            
            case EtinyNetLayerType::DENSE_LINEAR_DEPTHWISE: {
                if (layer_info.index < static_cast<int>(dlb_layers_.size())) {
                    const auto& layer = dlb_layers_[layer_info.index];
                    
                    // Dense blocks typically preserve spatial dimensions
                    int out_h = current_h;
                    int out_w = current_w;
                    int out_c = layer->linear_block.out_channels;
                    
                    layer_info.output_h = out_h;
                    layer_info.output_w = out_w;
                    layer_info.output_c = out_c;
                    
                    current_h = out_h;
                    current_w = out_w;
                    current_c = out_c;
                }
                break;
            }
            
            case EtinyNetLayerType::LINEAR_CLASSIFIER: {
                // Classifier outputs num_classes_ after global pooling
                layer_info.output_h = 1;
                layer_info.output_w = 1;
                layer_info.output_c = num_classes_;
                break;
            }
        }
    }
    
    // Debug output removed for clean inference
    
    // Suppress unused variable warning
    (void)current_c;
}

void EtinyNetEvaluator::allocate_working_buffers() {
    // Calculate the maximum buffer size needed for intermediate results
    size_t max_buffer_size = input_size_ * input_size_ * 3;  // Start with input size
    
    for (const auto& layer_info : layer_sequence_) {
        size_t layer_buffer_size = layer_info.output_h * layer_info.output_w * layer_info.output_c;
        max_buffer_size = std::max(max_buffer_size, layer_buffer_size);
    }
    
    // Allocate working buffers with some extra space for safety
    max_buffer_size = static_cast<size_t>(max_buffer_size * 1.2);  // 20% extra
    
    // Allocate final output buffer
    final_output_.resize(num_classes_);
    
    // Debug output removed for clean inference
}

void EtinyNetEvaluator::evaluate(const float* image_data, float* output, int image_h, int image_w) const {
    // Quantize input to int8
    int input_size = image_h * image_w * 3;
    std::vector<int8_t> quantized_input(input_size);
    quantize_input(image_data, quantized_input.data(), input_size);
    
    // Working buffers for intermediate results
    std::vector<int8_t> current_data = quantized_input;
    std::vector<int8_t> next_data;
    
    int current_h = image_h;
    int current_w = image_w;
    int current_c = 3;  // RGB input channels
    
    // Forward pass through all layers in sequence
    for (const auto& layer_info : layer_sequence_) {
        switch (layer_info.type) {
            case EtinyNetLayerType::STANDARD_CONV: {
                if (layer_info.index < static_cast<int>(conv_layers_.size())) {
                    const auto& layer = conv_layers_[layer_info.index];
                    
                    // Calculate output dimensions for standard convolution
                    // Assuming kernel=3, padding=1, stride=2 for initial conv
                    int stride = 2;  // EtinyNet initial conv has stride 2
                    int padding = 1;
                    int kernel = 3;
                    
                    int out_h = (current_h + 2 * padding - kernel) / stride + 1;
                    int out_w = (current_w + 2 * padding - kernel) / stride + 1;
                    int out_c = layer->out_channels;
                    
                    next_data.resize(out_h * out_w * out_c);
                    // ConvLayer expects float input, not quantized int8
                    layer->forward(image_data, next_data.data(), current_h, current_w, stride);
                    
                    current_data = std::move(next_data);
                    current_h = out_h;
                    current_w = out_w;
                    current_c = out_c;
                }
                break;
            }
            
            case EtinyNetLayerType::LINEAR_DEPTHWISE: {
                if (layer_info.index < static_cast<int>(lb_layers_.size())) {
                    const auto& layer = lb_layers_[layer_info.index];
                    
                    // Calculate output dimensions (stride and padding from layer)
                    int out_h = (current_h - 3 + 2) / layer->stride + 1;  // Assuming padding=1
                    int out_w = (current_w - 3 + 2) / layer->stride + 1;
                    int out_c = layer->out_channels;
                    
                    next_data.resize(out_h * out_w * out_c);
                    layer->forward(current_data.data(), next_data.data(), current_h, current_w, &buffer_pool_);
                    
                    current_data = std::move(next_data);
                    current_h = out_h;
                    current_w = out_w;
                    current_c = out_c;
                }
                break;
            }
            
            case EtinyNetLayerType::DENSE_LINEAR_DEPTHWISE: {
                if (layer_info.index < static_cast<int>(dlb_layers_.size())) {
                    const auto& layer = dlb_layers_[layer_info.index];
                    
                    // Output dimensions same as input for dense blocks (usually stride=1)
                    int out_h = current_h;  // Dense blocks typically preserve spatial dimensions
                    int out_w = current_w;
                    int out_c = layer->linear_block.out_channels;
                    
                    next_data.resize(out_h * out_w * out_c);
                    layer->forward(current_data.data(), next_data.data(), current_h, current_w, &buffer_pool_);
                    
                    current_data = std::move(next_data);
                    current_h = out_h;
                    current_w = out_w;
                    current_c = out_c;
                }
                break;
            }
            
            case EtinyNetLayerType::LINEAR_CLASSIFIER: {
                // Apply global average pooling before classifier
                std::vector<int8_t> pooled_features(current_c);
                apply_global_avg_pool(current_data.data(), pooled_features.data(), current_h, current_w, current_c);
                
                // Final classifier
                if (classifier_) {
                    classifier_->forward(pooled_features.data(), final_output_.data());
                }
                break;
            }
        }
    }
    
    // Copy to output
    for (int i = 0; i < num_classes_; ++i) {
        output[i] = final_output_[i];
    }
}

int EtinyNetEvaluator::predict(const float* image_data, int image_h, int image_w) const {
    std::vector<float> scores(num_classes_);
    evaluate(image_data, scores.data(), image_h, image_w);
    
    return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

std::pair<int, float> EtinyNetEvaluator::predict_with_confidence(const float* image_data, int image_h, int image_w) const {
    std::vector<float> scores(num_classes_);
    evaluate(image_data, scores.data(), image_h, image_w);
    
    int best_class = std::max_element(scores.begin(), scores.end()) - scores.begin();
    float confidence = scores[best_class];
    
    return std::make_pair(best_class, confidence);
}

void EtinyNetEvaluator::quantize_input(const float* input, int8_t* output, int size) const {
    constexpr float scale = 127.0f;  // Quantization scale
    for (int i = 0; i < size; ++i) {
        float val = input[i] * scale;
        output[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, val)));
    }
}

void EtinyNetEvaluator::apply_relu_inplace(int8_t* data, int size) const {
    for (int i = 0; i < size; ++i) {
        data[i] = std::max(static_cast<int8_t>(0), data[i]);
    }
}

void EtinyNetEvaluator::apply_global_avg_pool(const int8_t* input, int8_t* output, int h, int w, int channels) const {
    for (int c = 0; c < channels; ++c) {
        int32_t sum = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                sum += input[(y * w + x) * channels + c];
            }
        }
        int avg = sum / (h * w);
        output[c] = static_cast<int8_t>(std::max(-127, std::min(127, avg)));
    }
}

void EtinyNetEvaluator::add_skip_connection(const int8_t* input, int8_t* output, int size) const {
    for (int i = 0; i < size; ++i) {
        int result = static_cast<int32_t>(output[i]) + static_cast<int32_t>(input[i]);
        output[i] = static_cast<int8_t>(std::max(-127, std::min(127, result)));
    }
}



} // namespace nnue 