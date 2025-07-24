#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <mutex> // Added for MemoryPool mutex

namespace nnue {

// Default architecture constants (can be overridden by model file)
// Updated for 0.98M parameter target to match EtinyNet-0.98M
constexpr int DEFAULT_L1_SIZE = 1024;
constexpr int DEFAULT_L2_SIZE = 15;
constexpr int DEFAULT_L3_SIZE = 32;
constexpr int DEFAULT_INPUT_CHANNELS = 3;
constexpr int DEFAULT_OUTPUT_CHANNELS = 8;  // Reduced from 64 to match smaller feature set
constexpr int DEFAULT_CONV_KERNEL_SIZE = 3;
constexpr int DEFAULT_INPUT_IMAGE_SIZE = 96;
constexpr int DEFAULT_OUTPUT_GRID_SIZE = 10;  // Reduced from 32 to match smaller feature set

// Quantization constants
constexpr float FT_SCALE = 64.0f;
constexpr float HIDDEN_SCALE = 64.0f;
constexpr float OUTPUT_SCALE = 16.0f;
constexpr float NNUE2SCORE = 600.0f;
constexpr float QUANTIZED_ONE = 127.0f;

// SIMD alignment
constexpr int CACHE_LINE_SIZE = 64;

// Memory buffer pool for efficient temporary allocations
class MemoryPool {
private:
    struct Buffer {
        std::unique_ptr<int8_t[]> data;
        size_t size;
        bool in_use;
        
        Buffer(size_t s) : data(std::make_unique<int8_t[]>(s)), size(s), in_use(false) {}
    };
    
    mutable std::vector<std::unique_ptr<Buffer>> buffers_;
    mutable std::mutex mutex_;
    
public:
    // Get a buffer of at least the requested size
    int8_t* get_buffer(size_t min_size) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find existing buffer that's large enough and not in use
        for (auto& buf : buffers_) {
            if (!buf->in_use && buf->size >= min_size) {
                buf->in_use = true;
                return buf->data.get();
            }
        }
        
        // Create new buffer with some extra space to avoid frequent reallocations
        size_t alloc_size = std::max(min_size, size_t(4096));  // At least 4KB
        auto new_buffer = std::make_unique<Buffer>(alloc_size);
        int8_t* ptr = new_buffer->data.get();
        new_buffer->in_use = true;
        buffers_.push_back(std::move(new_buffer));
        
        return ptr;
    }
    
    // Return a buffer to the pool
    void return_buffer(int8_t* ptr) const {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& buf : buffers_) {
            if (buf->data.get() == ptr) {
                buf->in_use = false;
                break;
            }
        }
    }
    
    // RAII wrapper for automatic buffer management
    class BufferLock {
    private:
        const MemoryPool* pool_;
        int8_t* buffer_;
        
    public:
        BufferLock(const MemoryPool* pool, size_t size) : pool_(pool), buffer_(pool->get_buffer(size)) {}
        ~BufferLock() { pool_->return_buffer(buffer_); }
        
        int8_t* get() const { return buffer_; }
        
        // Delete copy constructor and assignment
        BufferLock(const BufferLock&) = delete;
        BufferLock& operator=(const BufferLock&) = delete;
        
        // Move constructor and assignment
        BufferLock(BufferLock&& other) noexcept : pool_(other.pool_), buffer_(other.buffer_) {
            other.buffer_ = nullptr;
        }
        BufferLock& operator=(BufferLock&& other) noexcept {
            if (this != &other) {
                if (buffer_) pool_->return_buffer(buffer_);
                pool_ = other.pool_;
                buffer_ = other.buffer_;
                other.buffer_ = nullptr;
            }
            return *this;
        }
    };
    
    BufferLock get_buffer_lock(size_t size) const {
        return BufferLock(this, size);
    }
};

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

// Dynamic grid for configurable feature spaces
struct DynamicGrid {
    int grid_size;
    int num_channels;
    std::vector<std::vector<uint64_t>> pixels;
    
    DynamicGrid(int grid_size, int num_channels) 
        : grid_size(grid_size), num_channels(num_channels) {
        pixels.resize(grid_size);
        for (auto& row : pixels) {
            row.resize(grid_size, 0);
        }
    }
    
    void clear() {
        for (auto& row : pixels) {
            std::fill(row.begin(), row.end(), 0);
        }
    }
    
    // Convert from int8_t conv output to uint64 grid (for small channel counts)
    // For larger channel counts, use direct indexing
    void from_conv_output(const int8_t* conv_data, float threshold = 0.0f) {
        clear();
        for (int h = 0; h < grid_size; ++h) {
            for (int w = 0; w < grid_size; ++w) {
                uint64_t pixel_features = 0;
                for (int c = 0; c < std::min(num_channels, 64); ++c) {
                    int feature_idx = (h * grid_size + w) * num_channels + c;
                    if (static_cast<float>(conv_data[feature_idx]) > threshold) {
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
        for (int h = 0; h < grid_size; ++h) {
            for (int w = 0; w < grid_size; ++w) {
                uint64_t pixel = pixels[h][w];
                if (pixel != 0 && num_channels <= 64) {
                    // Extract active bits efficiently for small channel counts
                    while (pixel) {
                        int bit_pos = __builtin_ctzll(pixel);  // Count trailing zeros
                        int feature_idx = (h * grid_size + w) * num_channels + bit_pos;
                        active_features.push_back(feature_idx);
                        pixel &= pixel - 1;  // Clear lowest set bit
                    }
                } else if (num_channels > 64) {
                    // For larger channel counts, check each channel individually
                    // (This is a fallback - for efficiency with large channel counts,
                    // consider using a different data structure)
                    for (int c = 0; c < num_channels; ++c) {
                        int feature_idx = (h * grid_size + w) * num_channels + c;
                        // Note: This needs the original conv data, so this path needs refactoring
                        // For now, assume we stick to <= 64 channels for uint64 efficiency
                        (void)feature_idx;  // Suppress unused variable warning
                    }
                }
            }
        }
    }
    
    // Count total active features
    int count_active_features() const {
        int total = 0;
        for (int h = 0; h < grid_size; ++h) {
            for (int w = 0; w < grid_size; ++w) {
                if (num_channels <= 64) {
                    total += __builtin_popcountll(pixels[h][w]);
                }
            }
        }
        return total;
    }
};

// Convolution layer for configurable input/output sizes
struct ConvLayer {
    AlignedVector<int8_t> weights;
    AlignedVector<int32_t> biases;
    float scale;
    
    // Architecture parameters
    int out_channels;
    int in_channels;
    int kernel_h;
    int kernel_w;
    
    ConvLayer();
    ~ConvLayer() = default;
    
    // Configurable convolution
    void forward(const float* input, int8_t* output, int input_h, int input_w, int stride) const;
    
    bool load_from_stream(std::ifstream& file);
};

// Feature transformer: converts sparse features to dense representation
struct FeatureTransformer {
    AlignedVector<int16_t> weights;
    AlignedVector<int32_t> biases;
    float scale;
    
    // Architecture parameters
    int num_features;
    int output_size;
    
    FeatureTransformer();
    ~FeatureTransformer() = default;
    
    // Forward pass with sparse input
    void forward(const std::vector<int>& active_features, int16_t* output) const;
    
    // Chess engine-style incremental updates (SIMD optimized)
    void add_feature(int feature_idx, int16_t* accumulator) const;
    void remove_feature(int feature_idx, int16_t* accumulator) const;
    void move_feature(int from_idx, int to_idx, int16_t* accumulator) const;
    
    // Efficient accumulator update (batch operations)
    void update_accumulator(const std::vector<int>& added_features,
                           const std::vector<int>& removed_features,
                           int16_t* accumulator) const;
    
    // SIMD-optimized implementations
    void add_feature_simd(int feature_idx, int16_t* accumulator) const;
    void remove_feature_simd(int feature_idx, int16_t* accumulator) const;
    void forward_simd(const std::vector<int>& active_features, int16_t* output) const;
    
    bool load_from_stream(std::ifstream& file);
};

// Dense layer stack for configurable L1->L2->L3->1
struct LayerStack {
    // Architecture parameters
    int l1_size;
    int l2_size;
    int l3_size;
    
    // L1 layer (combined l1 + l1_fact for main computation)
    AlignedVector<int8_t> l1_weights;
    AlignedVector<int32_t> l1_biases;
    float l1_scale;
    
    // L1 factorization layer (for computing factorization outputs)
    AlignedVector<int8_t> l1_fact_weights;
    AlignedVector<int32_t> l1_fact_biases;
    float l1_fact_scale;
    
    // L2 layer
    AlignedVector<int8_t> l2_weights;
    AlignedVector<int32_t> l2_biases;
    float l2_scale;
    
    // Output layer
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
    
    // Forward pass through all layers with layer stack index for bucket selection
    float forward(const int16_t* input, int layer_stack_index = 0) const;
    
    bool load_from_stream(std::ifstream& file);
};

// ===== EtinyNet Components =====
// (DepthwiseSeparableConv support was removed – EtinyNet now uses only Conv + LB/DLB)

// Linear Depthwise Block (LB) from EtinyNet paper
struct LinearDepthwiseBlock {
    // First depthwise conv (no activation after this)
    AlignedVector<int8_t> dconv1_weights;
    float dconv1_scale;
    
    // Pointwise conv (with activation)
    AlignedVector<int8_t> pconv_weights;
    AlignedVector<int32_t> pconv_biases;
    float pconv_scale;
    
    // Second depthwise conv (with activation)
    AlignedVector<int8_t> dconv2_weights;
    float dconv2_scale;
    
    // Final pointwise conv (output)
    AlignedVector<int8_t> pconv_out_weights;
    AlignedVector<int32_t> pconv_out_biases;
    float pconv_out_scale;
    
    // Architecture parameters
    int in_channels;
    int mid_channels;
    int out_channels;
    int stride;
    
    LinearDepthwiseBlock();
    ~LinearDepthwiseBlock() = default;
    
    // Forward pass: dconv1 -> pconv+ReLU -> dconv2+ReLU -> pconv_out
    void forward(const int8_t* input, int8_t* output, int input_h, int input_w, const MemoryPool* pool = nullptr) const;
    
    bool load_from_stream(std::ifstream& file);
};

// Dense Linear Depthwise Block (DLB) from EtinyNet paper
struct DenseLinearDepthwiseBlock {
    LinearDepthwiseBlock linear_block;
    bool use_skip_connection;
    
    DenseLinearDepthwiseBlock();
    ~DenseLinearDepthwiseBlock() = default;
    
    // Forward pass with optional skip connection
    void forward(const int8_t* input, int8_t* output, int input_h, int input_w, const MemoryPool* pool = nullptr) const;
    
    bool load_from_stream(std::ifstream& file);
};

// Linear classifier layer  
struct LinearLayer {
    AlignedVector<int8_t> weights;
    AlignedVector<int32_t> biases;
    float scale;
    
    int in_features;
    int out_features;
    
    LinearLayer();
    ~LinearLayer() = default;
    
    void forward(const int8_t* input, float* output) const;
    
    bool load_from_stream(std::ifstream& file);
};

// EtinyNet layer types for serialization
enum class EtinyNetLayerType {
    STANDARD_CONV = 0,
    LINEAR_DEPTHWISE = 1,
    DENSE_LINEAR_DEPTHWISE = 2,
    LINEAR_CLASSIFIER = 3
};

// Main EtinyNet evaluator
class EtinyNetEvaluator {
private:
    // Architecture metadata
    std::string variant_;
    int num_classes_;
    int input_size_;
    int conv_channels_;
    int final_channels_;
    bool use_asq_;
    int asq_bits_;
    float lambda_param_;
    
    // Network layers (stored in execution order)
    std::vector<std::unique_ptr<ConvLayer>> conv_layers_;
    // Removed ds_layers_ (depth-wise separable convs) – no longer used
    std::vector<std::unique_ptr<LinearDepthwiseBlock>> lb_layers_;
    std::vector<std::unique_ptr<DenseLinearDepthwiseBlock>> dlb_layers_;
    std::unique_ptr<LinearLayer> classifier_;
    
    // Layer execution sequence (stores layer type and index)
    struct LayerInfo {
        EtinyNetLayerType type;
        int index;
        int output_h, output_w, output_c;  // Output dimensions
    };
    std::vector<LayerInfo> layer_sequence_;
    
    // Working buffers (dynamically allocated based on model)
    mutable std::vector<AlignedVector<int8_t>> intermediate_buffers_;
    mutable AlignedVector<float> final_output_;
    mutable MemoryPool buffer_pool_;  // Memory pool for temporary allocations

public:
    EtinyNetEvaluator();
    ~EtinyNetEvaluator() = default;
    
    // Load model from .etiny file
    bool load_model(const std::string& path);
    
    // Evaluate image: RGB float[H*W*3] -> class scores
    void evaluate(const float* image_data, float* output, int image_h = 112, int image_w = 112) const;
    
    // Get top-1 prediction
    int predict(const float* image_data, int image_h = 112, int image_w = 112) const;
    
    // Get prediction with confidence
    std::pair<int, float> predict_with_confidence(const float* image_data, int image_h = 112, int image_w = 112) const;
    
    // Architecture info
    const std::string& get_variant() const { return variant_; }
    int get_num_classes() const { return num_classes_; }
    int get_input_size() const { return input_size_; }
    bool uses_asq() const { return use_asq_; }
    
private:
    // Helper functions
    void calculate_layer_dimensions();
    void allocate_working_buffers();
    bool load_header(std::ifstream& file);
    bool load_layers(std::ifstream& file);
    void quantize_input(const float* input, int8_t* output, int size) const;
    void apply_relu_inplace(int8_t* data, int size) const;
    void apply_global_avg_pool(const int8_t* input, int8_t* output, int h, int w, int channels) const;
    void add_skip_connection(const int8_t* input, int8_t* output, int size) const;
};

// ===== End EtinyNet Components =====

// Main NNUE evaluator with configurable architecture
class NNUEEvaluator {
private:
    ConvLayer conv_layer_;
    FeatureTransformer feature_transformer_;
    std::vector<LayerStack> layer_stacks_;
    
    // Architecture metadata (read from model file)
    int num_features_;
    int l1_size_;
    int l2_size_;
    int l3_size_;
    int num_ls_buckets_;
    int grid_size_;
    int num_channels_per_square_;
    float visual_threshold_;
    
    // Quantization parameters
    float nnue2score_;
    float quantized_one_;
    
    // Working buffers (dynamically sized)
    mutable AlignedVector<int8_t> conv_output_;
    mutable std::unique_ptr<DynamicGrid> feature_grid_;
    mutable AlignedVector<int16_t> ft_output_;
    mutable std::vector<int> active_features_;
    
    // Chess engine-style accumulator management
    mutable AlignedVector<int16_t> accumulator_;          // Persistent accumulator state
    mutable AlignedVector<int16_t> backup_accumulator_;   // For backup/restore
    mutable std::vector<int> last_active_features_;       // Track feature changes
    mutable bool accumulator_dirty_;                      // Needs full refresh
    mutable bool incremental_enabled_;                    // Enable incremental updates

public:
    NNUEEvaluator();
    ~NNUEEvaluator() = default;
    
    // Load model from .nnue file
    bool load_model(const std::string& path);
    
    // Evaluate image: RGB float[H*W*3] -> score
    float evaluate(const float* image_data, int image_h = 96, int image_w = 96, int layer_stack_index = 0) const;
    
    // Chess engine-style incremental evaluation
    float evaluate_incremental(const std::vector<int>& current_features, int layer_stack_index = 0) const;
    
    // Accumulator management (like chess engine)
    void save_accumulator() const;                        // Backup current state
    void restore_accumulator() const;                     // Restore from backup
    void refresh_accumulator(const std::vector<int>& features) const;  // Full refresh
    void update_features(const std::vector<int>& added, const std::vector<int>& removed) const;
    
    // Enable/disable incremental updates
    void enable_incremental(bool enable = true) const { incremental_enabled_ = enable; }
    void mark_dirty() const { accumulator_dirty_ = true; }
    
    // Get model info
    int get_num_layer_stacks() const { return num_ls_buckets_; }
    int get_num_features() const { return num_features_; }
    int get_l1_size() const { return l1_size_; }
    int get_l2_size() const { return l2_size_; }
    int get_l3_size() const { return l3_size_; }
    int get_grid_size() const { return grid_size_; }
    int get_num_channels_per_square() const { return num_channels_per_square_; }
    float get_visual_threshold() const { return visual_threshold_; }
    
    // Utility functions
    static std::vector<int> extract_features(const int8_t* grid_data, 
                                           int grid_size, int num_channels,
                                           float threshold = 0.0f) {
        std::vector<int> active_features;
        
        for (int h = 0; h < grid_size; ++h) {
            for (int w = 0; w < grid_size; ++w) {
                for (int c = 0; c < num_channels; ++c) {
                    int feature_idx = (h * grid_size + w) * num_channels + c;
                    if (static_cast<float>(grid_data[feature_idx]) > threshold) {
                        active_features.push_back(feature_idx);
                    }
                }
            }
        }
        
        return active_features;
    }
};

// Forward declarations for SIMD implementations
namespace simd {
    // Check CPU capabilities
    bool has_avx2();
    bool has_neon();
    
    // SIMD implementations will be in separate files
    void conv2d_unrolled_avx2(const float* input, const int8_t* weights,
                              const int32_t* biases, int8_t* output, float scale,
                              int input_h, int input_w, int out_channels, int stride);
    
    void conv2d_unrolled_neon(const float* input, const int8_t* weights,
                              const int32_t* biases, int8_t* output, float scale,
                              int input_h, int input_w, int out_channels, int stride);
                              
    void conv2d_unrolled_scalar(const float* input, const int8_t* weights,
                                const int32_t* biases, int8_t* output, float scale,
                                int input_h, int input_w, int out_channels, int stride);
    
    // Feature transformer SIMD operations
    void ft_forward_avx2(const std::vector<int>& features, const int16_t* weights,
                         const int32_t* biases, int16_t* output, int num_features,
                         int output_size, float scale);
                         
    void ft_forward_neon(const std::vector<int>& features, const int16_t* weights,
                         const int32_t* biases, int16_t* output, int num_features,
                         int output_size, float scale);
                         
    void ft_forward_scalar(const std::vector<int>& features, const int16_t* weights,
                           const int32_t* biases, int16_t* output, int num_features,
                           int output_size, float scale);
    
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
                             
    // Dense layer with int8_t input (for L2 layer)
    void dense_forward_scalar_int16(const int8_t* input, const int8_t* weights,
                                   const int32_t* biases, int8_t* output,
                                   int input_size, int output_size, float scale);
    
    // Chess engine-style incremental updates (SIMD optimized)
    void add_feature_avx2(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size);
    void remove_feature_avx2(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size);
    void add_feature_neon(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size);
    void remove_feature_neon(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size);
    void add_feature_scalar(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size);
    void remove_feature_scalar(int feature_idx, const int16_t* weights, int16_t* accumulator, int output_size);
}

} // namespace nnue
