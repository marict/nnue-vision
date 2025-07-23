#include "../include/nnue_engine.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace nnue {

// ConvLayer implementation
ConvLayer::ConvLayer() : scale(64.0f), out_channels(0), in_channels(0), kernel_h(0), kernel_w(0) {}

bool ConvLayer::load_from_stream(std::ifstream& file) {
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
    
    // Read weights
    int weight_count = out_channels * in_channels * kernel_h * kernel_w;
    weights.resize(weight_count);
    file.read(reinterpret_cast<char*>(weights.data()), weight_count);
    
    // Read bias dimensions and data
    uint32_t bias_count;
    file.read(reinterpret_cast<char*>(&bias_count), sizeof(uint32_t));
    if (bias_count != out_channels) {
        std::cerr << "Conv bias count mismatch" << std::endl;
        return false;
    }
    
    biases.resize(bias_count);
    file.read(reinterpret_cast<char*>(biases.data()), bias_count * sizeof(int32_t));
    
    return file.good();
}

void ConvLayer::forward(const float* input, int8_t* output, int input_h, int input_w, int stride) const {
    // Calculate output dimensions
    int output_h = (input_h + 2 - kernel_h) / stride + 1;  // With padding=1
    int output_w = (input_w + 2 - kernel_w) / stride + 1;
    
    // Unrolled convolution for efficiency
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
                            for (int in_c = 0; in_c < in_channels; ++in_c) {
                                int input_idx = (in_h * input_w + in_w) * in_channels + in_c;
                                int weight_idx = ((out_c * kernel_h + kh) * kernel_w + kw) * in_channels + in_c;
                                
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

// FeatureTransformer implementation
FeatureTransformer::FeatureTransformer() : scale(64.0f), num_features(0), output_size(0) {}

bool FeatureTransformer::load_from_stream(std::ifstream& file) {
    // Read scale
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    
    // Read dimensions
    file.read(reinterpret_cast<char*>(&num_features), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&output_size), sizeof(uint32_t));
    
    // Read weights
    int weight_count = num_features * output_size;
    weights.resize(weight_count);
    file.read(reinterpret_cast<char*>(weights.data()), weight_count * sizeof(int16_t));
    
    // Read bias dimensions and data
    uint32_t bias_count;
    file.read(reinterpret_cast<char*>(&bias_count), sizeof(uint32_t));
    if (bias_count != output_size) {
        std::cerr << "FT bias count mismatch" << std::endl;
        return false;
    }
    
    biases.resize(bias_count);
    file.read(reinterpret_cast<char*>(biases.data()), bias_count * sizeof(int32_t));
    
    return file.good();
}

void FeatureTransformer::forward(const std::vector<int>& active_features, int16_t* output) const {
    // Initialize with bias
    for (int i = 0; i < output_size; ++i) {
        output[i] = static_cast<int16_t>(biases[i]);
    }
    
    // Accumulate active features
    for (int feature_idx : active_features) {
        if (feature_idx >= 0 && feature_idx < num_features) {
            for (int i = 0; i < output_size; ++i) {
                int weight_idx = feature_idx * output_size + i;
                output[i] += weights[weight_idx];
            }
        }
    }
}

void FeatureTransformer::update_accumulator(const std::vector<int>& added_features,
                                           const std::vector<int>& removed_features,
                                           int16_t* accumulator) const {
    // Add new features
    for (int feature_idx : added_features) {
        if (feature_idx >= 0 && feature_idx < num_features) {
            for (int i = 0; i < output_size; ++i) {
                int weight_idx = feature_idx * output_size + i;
                accumulator[i] += weights[weight_idx];
            }
        }
    }
    
    // Remove old features
    for (int feature_idx : removed_features) {
        if (feature_idx >= 0 && feature_idx < num_features) {
            for (int i = 0; i < output_size; ++i) {
                int weight_idx = feature_idx * output_size + i;
                accumulator[i] -= weights[weight_idx];
            }
        }
    }
}

// LayerStack implementation
LayerStack::LayerStack() : l1_size(0), l2_size(0), l3_size(0), l1_scale(64.0f), l2_scale(64.0f), output_scale(16.0f) {}

bool LayerStack::load_from_stream(std::ifstream& file) {
    // Read scales
    file.read(reinterpret_cast<char*>(&l1_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&l2_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&output_scale), sizeof(float));
    
    // Read L1 layer
    uint32_t l1_out_size, l1_in_size;
    file.read(reinterpret_cast<char*>(&l1_out_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&l1_in_size), sizeof(uint32_t));
    
    l1_size = l1_in_size;
    l2_size = l1_out_size;
    
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
    
    if (l2_in_size != l2_size * 2) {
        std::cerr << "Invalid L2 input size: " << l2_in_size << " (expected " << l2_size * 2 << ")" << std::endl;
        return false;
    }
    
    l3_size = l2_out_size;
    
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
    
    if (out_in_size != l3_size || out_out_size != 1) {
        std::cerr << "Invalid output layer dimensions: " << out_in_size << " -> " << out_out_size << std::endl;
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
    // L1: l1_size -> l2_size with ClippedReLU
    std::vector<int8_t> l1_output(l2_size);
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::dense_forward_avx2(input, l1_weights.data(), l1_biases.data(), 
                                   l1_output.data(), l1_size, l2_size, l1_scale);
        } else
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::dense_forward_neon(input, l1_weights.data(), l1_biases.data(),
                                   l1_output.data(), l1_size, l2_size, l1_scale);
        } else
    #endif
        {
            simd::dense_forward_scalar(input, l1_weights.data(), l1_biases.data(),
                                     l1_output.data(), l1_size, l2_size, l1_scale);
        }
    
    // Apply squared concatenation: [squared(l1), original(l1)] to match Python model
    std::vector<int8_t> l1_expanded(l2_size * 2);
    for (int i = 0; i < l2_size; ++i) {
        // Squared part (first half) - multiply by (127/128) to match Python quantization
        int32_t squared = static_cast<int32_t>(l1_output[i]) * static_cast<int32_t>(l1_output[i]);
        squared = (squared * 127) / 128;  // Apply (127/128) factor
        l1_expanded[i] = static_cast<int8_t>(std::max(0, std::min(127, squared)));
        
        // Original part (second half)
        l1_expanded[i + l2_size] = l1_output[i];
    }
    
    // L2: (l2_size * 2) -> l3_size with ClippedReLU
    std::vector<int8_t> l2_output(l3_size);
    #ifdef __AVX2__
        if (simd::has_avx2()) {
            simd::dense_forward_avx2(reinterpret_cast<const int16_t*>(l1_expanded.data()), 
                                   l2_weights.data(), l2_biases.data(),
                                   l2_output.data(), l2_size * 2, l3_size, l2_scale);
        } else
    #endif
    #ifdef __ARM_NEON__
        if (simd::has_neon()) {
            simd::dense_forward_neon(reinterpret_cast<const int16_t*>(l1_expanded.data()),
                                   l2_weights.data(), l2_biases.data(),
                                   l2_output.data(), l2_size * 2, l3_size, l2_scale);
        } else
    #endif
        {
            simd::dense_forward_scalar(reinterpret_cast<const int16_t*>(l1_expanded.data()),
                                     l2_weights.data(), l2_biases.data(),
                                     l2_output.data(), l2_size * 2, l3_size, l2_scale);
        }
    
    // Output: l3_size -> 1 (no activation)
    int32_t output_acc = output_biases[0];
    for (int i = 0; i < l3_size; ++i) {
        output_acc += static_cast<int32_t>(l2_output[i]) * static_cast<int32_t>(output_weights[i]);
    }
    
    // Apply output scaling and convert to float
    return static_cast<float>(output_acc) / output_scale;
}

// NNUEEvaluator implementation
NNUEEvaluator::NNUEEvaluator() 
    : num_features_(0), l1_size_(0), l2_size_(0), l3_size_(0), 
      num_ls_buckets_(0), grid_size_(0), num_channels_per_square_(0),
      visual_threshold_(0.0f), nnue2score_(600.0f), quantized_one_(127.0f) {
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
    grid_size_ = static_cast<int>(std::sqrt(num_features / num_channels_per_square_));
    
    if (grid_size_ * grid_size_ * num_channels_per_square_ != num_features) {
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
    int conv_output_size = grid_size_ * grid_size_ * num_channels_per_square_;
    conv_output_.resize(conv_output_size);
    
    feature_grid_ = std::make_unique<DynamicGrid>(grid_size_, num_channels_per_square_);
    
    ft_output_.resize(l1_size_);
    
    std::cout << "Successfully loaded NNUE model:" << std::endl;
    std::cout << "  Features: " << num_features_ << " (" << grid_size_ << "x" << grid_size_ 
              << "x" << num_channels_per_square_ << ")" << std::endl;
    std::cout << "  Architecture: " << l1_size_ << " -> " << l2_size_ << " -> " << l3_size_ << " -> 1" << std::endl;
    std::cout << "  Layer stacks: " << num_ls_buckets_ << std::endl;
    std::cout << "  Visual threshold: " << visual_threshold_ << std::endl;
    
    return true;
}

float NNUEEvaluator::evaluate(const float* image_data, int image_h, int image_w, int layer_stack_index) const {
    if (layer_stack_index >= num_ls_buckets_) {
        layer_stack_index = 0;
    }
    
    // Calculate conv stride to hit target grid size
    int conv_stride = image_h / grid_size_;
    if (conv_stride < 1) conv_stride = 1;
    
    // Step 1: Convolution image_h x image_w x 3 -> grid_size x grid_size x num_channels_per_square
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
    float raw_output = layer_stacks_[layer_stack_index].forward(ft_output_.data());
    
    // Step 6: Apply final scaling
    return raw_output * nnue2score_;
}

std::vector<int> NNUEEvaluator::extract_features(const int8_t* grid_data, 
                                                int grid_size, int num_channels,
                                                float threshold) {
    std::vector<int> active_features;
    
    for (int h = 0; h < grid_size; ++h) {
        for (int w = 0; w < grid_size; ++w) {
            for (int c = 0; c < num_channels; ++c) {
                int idx = (h * grid_size + w) * num_channels + c;
                if (static_cast<float>(grid_data[idx]) > threshold) {
                    int feature_idx = (h * grid_size + w) * num_channels + c;
                    active_features.push_back(feature_idx);
                }
            }
        }
    }
    
    return active_features;
}

} // namespace nnue 