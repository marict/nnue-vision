#include "../include/nnue_engine.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace nnue {

// ConvLayer implementation
ConvLayer::ConvLayer() : scale(1.0f) {}

bool ConvLayer::load_from_stream(std::ifstream& file) {
    // Read scale
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    
    // Read weight dimensions
    uint32_t out_channels, in_channels, kernel_h, kernel_w;
    file.read(reinterpret_cast<char*>(&out_channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&in_channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&kernel_h), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&kernel_w), sizeof(uint32_t));
    
    if (out_channels != OUTPUT_CHANNELS || in_channels != INPUT_CHANNELS ||
        kernel_h != CONV_KERNEL_SIZE || kernel_w != CONV_KERNEL_SIZE) {
        std::cerr << "Invalid conv layer dimensions" << std::endl;
        return false;
    }
    
    // Read weights
    size_t weight_count = out_channels * in_channels * kernel_h * kernel_w;
    weights.resize(weight_count);
    file.read(reinterpret_cast<char*>(weights.data()), weight_count);
    
    // Read bias dimensions and data
    uint32_t bias_count;
    file.read(reinterpret_cast<char*>(&bias_count), sizeof(uint32_t));
    if (bias_count != out_channels) {
        std::cerr << "Invalid bias count" << std::endl;
        return false;
    }
    
    biases.resize(bias_count);
    file.read(reinterpret_cast<char*>(biases.data()), bias_count * sizeof(int32_t));
    
    return file.good();
}

void ConvLayer::forward(const float* input, int8_t* output) const {
    // Use SIMD optimized version if available
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::conv2d_unrolled_avx2(input, weights.data(), biases.data(), output, scale);
            return;
        }
    #endif
    
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::conv2d_unrolled_neon(input, weights.data(), biases.data(), output, scale);
            return;
        }
    #endif
    
    // Fallback to scalar implementation
    simd::conv2d_unrolled_scalar(input, weights.data(), biases.data(), output, scale);
}

// FeatureTransformer implementation
FeatureTransformer::FeatureTransformer() : scale(FT_SCALE) {}

bool FeatureTransformer::load_from_stream(std::ifstream& file) {
    // Read scale
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    
    // Read weight dimensions
    uint32_t num_features, output_size;
    file.read(reinterpret_cast<char*>(&num_features), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&output_size), sizeof(uint32_t));
    
    if (num_features != GRID_FEATURES || output_size != L1_SIZE) {
        std::cerr << "Invalid feature transformer dimensions" << std::endl;
        return false;
    }
    
    // Read weights
    size_t weight_count = num_features * output_size;
    weights.resize(weight_count);
    file.read(reinterpret_cast<char*>(weights.data()), weight_count * sizeof(int16_t));
    
    // Read bias dimensions and data
    uint32_t bias_count;
    file.read(reinterpret_cast<char*>(&bias_count), sizeof(uint32_t));
    if (bias_count != output_size) {
        std::cerr << "Invalid bias count" << std::endl;
        return false;
    }
    
    biases.resize(bias_count);
    file.read(reinterpret_cast<char*>(biases.data()), bias_count * sizeof(int32_t));
    
    return file.good();
}

void FeatureTransformer::forward(const std::vector<int>& active_features, int16_t* output) const {
    // Use SIMD optimized version if available
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::ft_forward_avx2(active_features, weights.data(), biases.data(), output, scale);
            return;
        }
    #endif
    
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::ft_forward_neon(active_features, weights.data(), biases.data(), output, scale);
            return;
        }
    #endif
    
    // Fallback to scalar implementation
    simd::ft_forward_scalar(active_features, weights.data(), biases.data(), output, scale);
}

void FeatureTransformer::update_accumulator(const std::vector<int>& added_features,
                                           const std::vector<int>& removed_features,
                                           int16_t* accumulator) const {
    // Remove features
    for (int feature_idx : removed_features) {
        const int16_t* feature_weights = weights.data() + feature_idx * L1_SIZE;
        for (int i = 0; i < L1_SIZE; ++i) {
            accumulator[i] -= feature_weights[i];
        }
    }
    
    // Add features
    for (int feature_idx : added_features) {
        const int16_t* feature_weights = weights.data() + feature_idx * L1_SIZE;
        for (int i = 0; i < L1_SIZE; ++i) {
            accumulator[i] += feature_weights[i];
        }
    }
}

// LayerStack implementation
LayerStack::LayerStack() : l1_scale(HIDDEN_SCALE), l2_scale(HIDDEN_SCALE), output_scale(OUTPUT_SCALE) {}

bool LayerStack::load_from_stream(std::ifstream& file) {
    // Read scales
    file.read(reinterpret_cast<char*>(&l1_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&l2_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&output_scale), sizeof(float));
    
    // Read L1 layer
    uint32_t l1_out_size, l1_in_size;
    file.read(reinterpret_cast<char*>(&l1_out_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l1_in_size), sizeof(uint32_t));
    
    if (l1_in_size != L1_SIZE || l1_out_size != L2_SIZE) {
        std::cerr << "Invalid L1 layer dimensions" << std::endl;
        return false;
    }
    
    l1_weights.resize(l1_out_size * l1_in_size);
    file.read(reinterpret_cast<char*>(l1_weights.data()), l1_weights.size());
    
    uint32_t l1_bias_count;
    file.read(reinterpret_cast<char*>(&l1_bias_count), sizeof(uint32_t));
    l1_biases.resize(l1_bias_count);
    file.read(reinterpret_cast<char*>(l1_biases.data()), l1_bias_count * sizeof(int32_t));
    
    // Read L2 layer
    uint32_t l2_out_size, l2_in_size;
    file.read(reinterpret_cast<char*>(&l2_out_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l2_in_size), sizeof(uint32_t));
    
    if (l2_in_size != L2_SIZE || l2_out_size != L3_SIZE) {
        std::cerr << "Invalid L2 layer dimensions" << std::endl;
        return false;
    }
    
    l2_weights.resize(l2_out_size * l2_in_size);
    file.read(reinterpret_cast<char*>(l2_weights.data()), l2_weights.size());
    
    uint32_t l2_bias_count;
    file.read(reinterpret_cast<char*>(&l2_bias_count), sizeof(uint32_t));
    l2_biases.resize(l2_bias_count);
    file.read(reinterpret_cast<char*>(l2_biases.data()), l2_bias_count * sizeof(int32_t));
    
    // Read output layer
    uint32_t out_out_size, out_in_size;
    file.read(reinterpret_cast<char*>(&out_out_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&out_in_size), sizeof(uint32_t));
    
    if (out_in_size != L3_SIZE || out_out_size != 1) {
        std::cerr << "Invalid output layer dimensions" << std::endl;
        return false;
    }
    
    output_weights.resize(out_out_size * out_in_size);
    file.read(reinterpret_cast<char*>(output_weights.data()), output_weights.size());
    
    uint32_t out_bias_count;
    file.read(reinterpret_cast<char*>(&out_bias_count), sizeof(uint32_t));
    output_biases.resize(out_bias_count);
    file.read(reinterpret_cast<char*>(output_biases.data()), out_bias_count * sizeof(int32_t));
    
    return file.good();
}

float LayerStack::forward(const int16_t* input) const {
    // L1: 3072 -> 15 with ClippedReLU
    int8_t l1_output[L2_SIZE];
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::dense_forward_avx2(input, l1_weights.data(), l1_biases.data(), 
                                   l1_output, L1_SIZE, L2_SIZE, l1_scale);
        } else
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::dense_forward_neon(input, l1_weights.data(), l1_biases.data(),
                                   l1_output, L1_SIZE, L2_SIZE, l1_scale);
        } else
    #endif
        {
            simd::dense_forward_scalar(input, l1_weights.data(), l1_biases.data(),
                                     l1_output, L1_SIZE, L2_SIZE, l1_scale);
        }
    
    // L2: 15 -> 32 with ClippedReLU
    int8_t l2_output[L3_SIZE];
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::dense_forward_avx2(reinterpret_cast<const int16_t*>(l1_output), 
                                   l2_weights.data(), l2_biases.data(),
                                   l2_output, L2_SIZE, L3_SIZE, l2_scale);
        } else
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::dense_forward_neon(reinterpret_cast<const int16_t*>(l1_output),
                                   l2_weights.data(), l2_biases.data(),
                                   l2_output, L2_SIZE, L3_SIZE, l2_scale);
        } else
    #endif
        {
            simd::dense_forward_scalar(reinterpret_cast<const int16_t*>(l1_output),
                                     l2_weights.data(), l2_biases.data(),
                                     l2_output, L2_SIZE, L3_SIZE, l2_scale);
        }
    
    // Output: 32 -> 1 (no activation)
    int32_t output_acc = output_biases[0];
    for (int i = 0; i < L3_SIZE; ++i) {
        output_acc += static_cast<int32_t>(l2_output[i]) * static_cast<int32_t>(output_weights[i]);
    }
    
    // Apply output scaling and convert to float
    return static_cast<float>(output_acc) / output_scale;
}

// NNUEEvaluator implementation
NNUEEvaluator::NNUEEvaluator() 
    : num_ls_buckets_(8), visual_threshold_(0.0f),
      conv_output_(OUTPUT_GRID_SIZE * OUTPUT_GRID_SIZE * OUTPUT_CHANNELS),
      ft_output_(L1_SIZE) {
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
    
    if (num_features != GRID_FEATURES || l1_size != L1_SIZE || 
        l2_size != L2_SIZE || l3_size != L3_SIZE) {
        std::cerr << "Architecture mismatch" << std::endl;
        return false;
    }
    
    // Read quantization parameters
    float nnue2score, quantized_one;
    file.read(reinterpret_cast<char*>(&nnue2score), sizeof(float));
    file.read(reinterpret_cast<char*>(&quantized_one), sizeof(float));
    file.read(reinterpret_cast<char*>(&visual_threshold_), sizeof(float));
    
    // Load conv layer
    if (!conv_layer_.load_from_stream(file)) {
        std::cerr << "Failed to load conv layer" << std::endl;
        return false;
    }
    
    // Load feature transformer
    if (!feature_transformer_.load_from_stream(file)) {
        std::cerr << "Failed to load feature transformer" << std::endl;
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
    }
    
    std::cout << "Successfully loaded NNUE model:" << std::endl;
    std::cout << "  Features: " << num_features << std::endl;
    std::cout << "  Layer stacks: " << num_ls_buckets_ << std::endl;
    std::cout << "  Visual threshold: " << visual_threshold_ << std::endl;
    
    return true;
}

float NNUEEvaluator::evaluate(const float* image_data, int layer_stack_index) const {
    if (layer_stack_index >= num_ls_buckets_) {
        layer_stack_index = 0;
    }
    
    // Step 1: Convolution 96x96x3 -> 32x32x64
    conv_layer_.forward(image_data, conv_output_.data());
    
    // Step 2: Convert to uint64 grid and extract active features efficiently
    feature_grid_.from_conv_output(conv_output_.data(), visual_threshold_);
    feature_grid_.extract_features(active_features_);
    
    // Step 3: Feature transformer (sparse -> dense)
    feature_transformer_.forward(active_features_, ft_output_.data());
    
    // Step 4: Apply clipped ReLU to feature transformer output
    for (int i = 0; i < L1_SIZE; ++i) {
        ft_output_[i] = std::max(static_cast<int16_t>(0), 
                                std::min(ft_output_[i], static_cast<int16_t>(QUANTIZED_ONE)));
    }
    
    // Step 5: Pass through layer stack
    float raw_output = layer_stacks_[layer_stack_index].forward(ft_output_.data());
    
    // Step 6: Apply final scaling
    return raw_output * NNUE2SCORE;
}

std::vector<int> NNUEEvaluator::extract_features(const int8_t* grid_data, float threshold) {
    std::vector<int> features;
    for (int i = 0; i < GRID_FEATURES; ++i) {
        if (static_cast<float>(grid_data[i]) > threshold) {
            features.push_back(i);
        }
    }
    return features;
}

} // namespace nnue 