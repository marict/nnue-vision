#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <fstream>

namespace nnue {

// Architecture constants matching the Python implementation
constexpr int L1_SIZE = 512;  // Increased from 256 for larger feature space
constexpr int L2_SIZE = 15;
constexpr int L3_SIZE = 32;
constexpr int INPUT_CHANNELS = 3;
constexpr int OUTPUT_CHANNELS = 64;  // 64 channels per pixel for uint64 efficiency
constexpr int CONV_KERNEL_SIZE = 3;
constexpr int INPUT_IMAGE_SIZE = 96;
constexpr int OUTPUT_GRID_SIZE = 32;  // Increased from 8 for fine spatial resolution
constexpr int GRID_FEATURES = OUTPUT_GRID_SIZE * OUTPUT_GRID_SIZE * OUTPUT_CHANNELS; // 65,536

// Quantization constants
constexpr float FT_SCALE = 64.0f;
constexpr float HIDDEN_SCALE = 64.0f;
constexpr float OUTPUT_SCALE = 16.0f;
constexpr float NNUE2SCORE = 600.0f;
constexpr float QUANTIZED_ONE = 127.0f;

// SIMD alignment
constexpr int CACHE_LINE_SIZE = 64;

// Forward declarations
struct ConvLayer;
struct FeatureTransformer;
struct LayerStack;
class NNUEEvaluator;

// Aligned memory allocation
template<typename T>
T* aligned_alloc(size_t count) {
    void* ptr = nullptr;
    #ifdef _WIN32
        ptr = _aligned_malloc(count * sizeof(T), CACHE_LINE_SIZE);
    #else
        if (posix_memalign(&ptr, CACHE_LINE_SIZE, count * sizeof(T)) != 0) {
            ptr = nullptr;
        }
    #endif
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_free(T* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// RAII wrapper for aligned memory
template<typename T>
class AlignedVector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;

public:
    AlignedVector() : data_(nullptr), size_(0), capacity_(0) {}
    
    explicit AlignedVector(size_t size) : size_(size), capacity_(size) {
        data_ = aligned_alloc<T>(capacity_);
        if (!data_) throw std::bad_alloc();
    }
    
    ~AlignedVector() {
        if (data_) aligned_free(data_);
    }
    
    // Move constructor
    AlignedVector(AlignedVector&& other) noexcept 
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    // Move assignment
    AlignedVector& operator=(AlignedVector&& other) noexcept {
        if (this != &other) {
            if (data_) aligned_free(data_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Delete copy constructor and assignment
    AlignedVector(const AlignedVector&) = delete;
    AlignedVector& operator=(const AlignedVector&) = delete;
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    
    void resize(size_t new_size) {
        if (new_size > capacity_) {
            T* new_data = aligned_alloc<T>(new_size);
            if (!new_data) throw std::bad_alloc();
            if (data_) {
                std::copy(data_, data_ + size_, new_data);
                aligned_free(data_);
            }
            data_ = new_data;
            capacity_ = new_size;
        }
        size_ = new_size;
    }
};

// Efficient uint64 grid for 32x32x64 features (1 uint64 per pixel)
struct Grid32x32 {
    uint64_t pixels[OUTPUT_GRID_SIZE][OUTPUT_GRID_SIZE];
    
    Grid32x32() { clear(); }
    
    void clear() {
        std::memset(pixels, 0, sizeof(pixels));
    }
    
    // Convert from int8_t conv output to uint64 grid
    void from_conv_output(const int8_t* conv_data, float threshold = 0.0f) {
        clear();
        for (int h = 0; h < OUTPUT_GRID_SIZE; ++h) {
            for (int w = 0; w < OUTPUT_GRID_SIZE; ++w) {
                uint64_t pixel_features = 0;
                for (int c = 0; c < OUTPUT_CHANNELS; ++c) {
                    int idx = (h * OUTPUT_GRID_SIZE + w) * OUTPUT_CHANNELS + c;
                    if (static_cast<float>(conv_data[idx]) > threshold) {
                        pixel_features |= (1ULL << c);
                    }
                }
                pixels[h][w] = pixel_features;
            }
        }
    }
    
    // Extract sparse feature indices efficiently
    void extract_features(std::vector<int>& active_features) const {
        active_features.clear();
        for (int h = 0; h < OUTPUT_GRID_SIZE; ++h) {
            for (int w = 0; w < OUTPUT_GRID_SIZE; ++w) {
                uint64_t pixel = pixels[h][w];
                if (pixel != 0) {
                    // Extract active bits efficiently
                    while (pixel) {
                        int bit_pos = __builtin_ctzll(pixel);  // Count trailing zeros
                        int feature_idx = (h * OUTPUT_GRID_SIZE + w) * OUTPUT_CHANNELS + bit_pos;
                        active_features.push_back(feature_idx);
                        pixel &= pixel - 1;  // Clear lowest set bit
                    }
                }
            }
        }
    }
    
    // Count total active features
    int count_active_features() const {
        int total = 0;
        for (int h = 0; h < OUTPUT_GRID_SIZE; ++h) {
            for (int w = 0; w < OUTPUT_GRID_SIZE; ++w) {
                total += __builtin_popcountll(pixels[h][w]);
            }
        }
        return total;
    }
    
    // Get neighborhood features (useful for spatial operations)
    uint64_t get_neighborhood_features(int h, int w, int radius = 1) const {
        uint64_t combined = 0;
        for (int dh = -radius; dh <= radius; ++dh) {
            for (int dw = -radius; dw <= radius; ++dw) {
                int nh = h + dh;
                int nw = w + dw;
                if (nh >= 0 && nh < OUTPUT_GRID_SIZE && nw >= 0 && nw < OUTPUT_GRID_SIZE) {
                    combined |= pixels[nh][nw];
                }
            }
        }
        return combined;
    }
};

// Convolution layer for 96x96x3 -> 32x32x64 downsampling (uint64 per pixel)
struct ConvLayer {
    AlignedVector<int8_t> weights;     // [64, 3, 3, 3] quantized weights
    AlignedVector<int32_t> biases;     // [64] quantized biases
    float scale;
    
    ConvLayer();
    ~ConvLayer() = default;
    
    // Unrolled convolution with stride=3
    void forward(const float* input, int8_t* output) const;
    
    bool load_from_stream(std::ifstream& file);
};

// Feature transformer: converts sparse 32x32x64 binary features to dense L1
struct FeatureTransformer {
    AlignedVector<int16_t> weights;    // [65536, 512] quantized weights
    AlignedVector<int32_t> biases;     // [512] quantized biases
    float scale;
    
    FeatureTransformer();
    ~FeatureTransformer() = default;
    
    // Forward pass with sparse input
    void forward(const std::vector<int>& active_features, int16_t* output) const;
    
    // Efficient accumulator update
    void update_accumulator(const std::vector<int>& added_features,
                           const std::vector<int>& removed_features,
                           int16_t* accumulator) const;
    
    bool load_from_stream(std::ifstream& file);
};

// Dense layer stack for L1->L2->L3->1
struct LayerStack {
    // L1 layer (512 -> 15)
    AlignedVector<int8_t> l1_weights;
    AlignedVector<int32_t> l1_biases;
    float l1_scale;
    
    // L2 layer (15 -> 32)
    AlignedVector<int8_t> l2_weights;
    AlignedVector<int32_t> l2_biases;
    float l2_scale;
    
    // Output layer (32 -> 1)
    AlignedVector<int8_t> output_weights;
    AlignedVector<int32_t> output_biases;
    float output_scale;
    
    LayerStack();
    ~LayerStack() = default;
    
    // Make it movable but not copyable
    LayerStack(const LayerStack&) = delete;
    LayerStack& operator=(const LayerStack&) = delete;
    LayerStack(LayerStack&&) = default;
    LayerStack& operator=(LayerStack&&) = default;
    
    // Forward pass through all layers
    float forward(const int16_t* input) const;
    
    bool load_from_stream(std::ifstream& file);
};

// Main NNUE evaluator
class NNUEEvaluator {
private:
    ConvLayer conv_layer_;
    FeatureTransformer feature_transformer_;
    std::vector<LayerStack> layer_stacks_;
    
    // Metadata
    int num_ls_buckets_;
    float visual_threshold_;
    
    // Working buffers
    mutable AlignedVector<int8_t> conv_output_;
    mutable Grid32x32 feature_grid_;         // Efficient uint64 grid representation
    mutable AlignedVector<int16_t> ft_output_;
    mutable std::vector<int> active_features_;

public:
    NNUEEvaluator();
    ~NNUEEvaluator() = default;
    
    // Load model from .nnue file
    bool load_model(const std::string& path);
    
    // Evaluate image: RGB float[96*96*3] -> score
    float evaluate(const float* image_data, int layer_stack_index = 0) const;
    
    // Get model info
    int get_num_layer_stacks() const { return num_ls_buckets_; }
    float get_visual_threshold() const { return visual_threshold_; }
    
    // Utility functions
    static std::vector<int> extract_features(const int8_t* grid_data, 
                                           float threshold = 0.0f);
};

// Utility functions for SIMD operations
namespace simd {
    // Check CPU capabilities
    bool has_avx2();
    bool has_neon();
    
    // SIMD implementations will be in separate files
    void conv2d_unrolled_avx2(const float* input, const int8_t* weights,
                              const int32_t* biases, int8_t* output, float scale);
    
    void conv2d_unrolled_neon(const float* input, const int8_t* weights,
                              const int32_t* biases, int8_t* output, float scale);
                              
    void conv2d_unrolled_scalar(const float* input, const int8_t* weights,
                                const int32_t* biases, int8_t* output, float scale);
    
    // Feature transformer SIMD operations
    void ft_forward_avx2(const std::vector<int>& features, const int16_t* weights,
                         const int32_t* biases, int16_t* output, float scale);
                         
    void ft_forward_neon(const std::vector<int>& features, const int16_t* weights,
                         const int32_t* biases, int16_t* output, float scale);
                         
    void ft_forward_scalar(const std::vector<int>& features, const int16_t* weights,
                           const int32_t* biases, int16_t* output, float scale);
    
    // Dense layer SIMD operations
    void dense_forward_avx2(const int16_t* input, const int8_t* weights,
                           const int32_t* biases, int8_t* output, 
                           int input_size, int output_size, float scale);
                           
    void dense_forward_neon(const int16_t* input, const int8_t* weights,
                           const int32_t* biases, int8_t* output,
                           int input_size, int output_size, float scale);
                           
    void dense_forward_scalar(const int16_t* input, const int8_t* weights,
                             const int32_t* biases, int8_t* output,
                             int input_size, int output_size, float scale);
}

} // namespace nnue
