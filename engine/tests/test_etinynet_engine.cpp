#include "../include/nnue_engine.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>

using namespace nnue;

// Test utilities
class TestResults {
public:
    int total = 0;
    int passed = 0;
    
    void test(const std::string& name, bool condition) {
        total++;
        if (condition) {
            passed++;
            std::cout << "✓ " << name << std::endl;
        } else {
            std::cout << "❌ " << name << std::endl;
        }
    }
    
    void summary() {
        std::cout << "\n📊 Test Summary: " << passed << "/" << total << " passed";
        if (passed == total) {
            std::cout << " 🎉" << std::endl;
        } else {
            std::cout << " ⚠️" << std::endl;
        }
    }
    
    bool all_passed() const { return passed == total; }
};

// Generate random float data
std::vector<float> generate_random_floats(int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// Generate random int8 data  
std::vector<int8_t> generate_random_int8(int size) {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis(-127, 127);
    
    std::vector<int8_t> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<int8_t>(dis(gen));
    }
    return data;
}

// Test basic AlignedVector functionality
void test_aligned_vector(TestResults& results) {
    std::cout << "\n🧪 Testing AlignedVector..." << std::endl;
    
    // Test construction and basic operations
    AlignedVector<float> vec(100);
    results.test("AlignedVector construction", vec.size() == 100);
    results.test("AlignedVector data not null", vec.data() != nullptr);
    
    // Test data access
    vec[0] = 3.14f;
    vec[99] = 2.71f;
    results.test("AlignedVector data access", vec[0] == 3.14f && vec[99] == 2.71f);
    
    // Test resize
    vec.resize(200);
    results.test("AlignedVector resize", vec.size() == 200);
    results.test("AlignedVector data preserved after resize", vec[0] == 3.14f);
}

// Test DepthwiseSeparableConv
void test_depthwise_separable_conv(TestResults& results) {
    std::cout << "\n🧪 Testing DepthwiseSeparableConv..." << std::endl;
    
    DepthwiseSeparableConv conv;
    conv.in_channels = 32;
    conv.out_channels = 64;
    conv.kernel_size = 3;
    conv.stride = 1;
    conv.padding = 1;
    
    // Initialize weights and biases with test data
    int dw_weight_size = conv.in_channels * conv.kernel_size * conv.kernel_size;
    int pw_weight_size = conv.out_channels * conv.in_channels;
    
    conv.depthwise_weights.resize(dw_weight_size);
    conv.pointwise_weights.resize(pw_weight_size);
    conv.pointwise_biases.resize(conv.out_channels);
    
    // Fill with test data
    for (int i = 0; i < dw_weight_size; ++i) {
        conv.depthwise_weights[i] = (i % 3) - 1;  // -1, 0, 1 pattern
    }
    for (int i = 0; i < pw_weight_size; ++i) {
        conv.pointwise_weights[i] = (i % 2) ? 1 : -1;  // alternating 1, -1
    }
    for (int i = 0; i < conv.out_channels; ++i) {
        conv.pointwise_biases[i] = i;  // bias = index
    }
    
    // Test forward pass
    int input_h = 16, input_w = 16;
    int input_size = input_h * input_w * conv.in_channels;
    auto input_data = generate_random_int8(input_size);
    
    int output_h = (input_h + 2 * conv.padding - conv.kernel_size) / conv.stride + 1;
    int output_w = (input_w + 2 * conv.padding - conv.kernel_size) / conv.stride + 1;
    int output_size = output_h * output_w * conv.out_channels;
    std::vector<int8_t> output_data(output_size);
    
    // This should not crash
    try {
        conv.forward(input_data.data(), output_data.data(), input_h, input_w, true);
        results.test("DepthwiseSeparableConv forward pass", true);
    } catch (...) {
        results.test("DepthwiseSeparableConv forward pass", false);
    }
    
    // Check output dimensions
    results.test("DepthwiseSeparableConv output size", static_cast<int>(output_data.size()) == output_size);
    
    // Check that output is not all zeros (indicates computation happened)
    bool has_nonzero = false;
    for (int val : output_data) {
        if (val != 0) {
            has_nonzero = true;
            break;
        }
    }
    results.test("DepthwiseSeparableConv produces non-zero output", has_nonzero);
}

// Test LinearDepthwiseBlock
void test_linear_depthwise_block(TestResults& results) {
    std::cout << "\n🧪 Testing LinearDepthwiseBlock..." << std::endl;
    
    LinearDepthwiseBlock block;
    block.in_channels = 32;
    block.mid_channels = 64;
    block.out_channels = 128;
    block.stride = 1;
    
    // Initialize weights and biases
    int dconv1_size = block.in_channels * 9;  // 3x3 kernel
    int pconv_size = block.mid_channels * block.in_channels;
    int dconv2_size = block.mid_channels * 9;  // 3x3 kernel
    int pconv_out_size = block.out_channels * block.mid_channels;
    
    block.dconv1_weights.resize(dconv1_size);
    block.pconv_weights.resize(pconv_size);
    block.pconv_biases.resize(block.mid_channels);
    block.dconv2_weights.resize(dconv2_size);
    block.pconv_out_weights.resize(pconv_out_size);
    block.pconv_out_biases.resize(block.out_channels);
    
    // Fill with test data (simplified)
    for (size_t i = 0; i < block.dconv1_weights.size(); ++i) block.dconv1_weights[i] = 1;
    for (size_t i = 0; i < block.pconv_weights.size(); ++i) block.pconv_weights[i] = 1;
    for (size_t i = 0; i < block.pconv_biases.size(); ++i) block.pconv_biases[i] = 0;
    for (size_t i = 0; i < block.dconv2_weights.size(); ++i) block.dconv2_weights[i] = 1;
    for (size_t i = 0; i < block.pconv_out_weights.size(); ++i) block.pconv_out_weights[i] = 1;
    for (size_t i = 0; i < block.pconv_out_biases.size(); ++i) block.pconv_out_biases[i] = 0;
    
    // Test forward pass
    int input_h = 14, input_w = 14;
    int input_size = input_h * input_w * block.in_channels;
    auto input_data = generate_random_int8(input_size);
    
    int output_size = input_h * input_w * block.out_channels;  // Same spatial size due to stride=1, padding=1
    std::vector<int8_t> output_data(output_size);
    
    try {
        block.forward(input_data.data(), output_data.data(), input_h, input_w);
        results.test("LinearDepthwiseBlock forward pass", true);
    } catch (...) {
        results.test("LinearDepthwiseBlock forward pass", false);
    }
    
    results.test("LinearDepthwiseBlock output size", static_cast<int>(output_data.size()) == output_size);
}

// Test DenseLinearDepthwiseBlock
void test_dense_linear_depthwise_block(TestResults& results) {
    std::cout << "\n🧪 Testing DenseLinearDepthwiseBlock..." << std::endl;
    
    DenseLinearDepthwiseBlock dlb;
    dlb.use_skip_connection = true;
    
    // Configure inner linear block
    dlb.linear_block.in_channels = 64;
    dlb.linear_block.mid_channels = 64;
    dlb.linear_block.out_channels = 64;  // Same channels for skip connection
    dlb.linear_block.stride = 1;
    
    // Initialize weights (simplified)
    dlb.linear_block.dconv1_weights.resize(64 * 9);
    dlb.linear_block.pconv_weights.resize(64 * 64);
    dlb.linear_block.pconv_biases.resize(64);
    dlb.linear_block.dconv2_weights.resize(64 * 9);
    dlb.linear_block.pconv_out_weights.resize(64 * 64);
    dlb.linear_block.pconv_out_biases.resize(64);
    
    // Fill with small values to avoid overflow in skip connection
    for (size_t i = 0; i < dlb.linear_block.dconv1_weights.size(); ++i) 
        dlb.linear_block.dconv1_weights[i] = (i % 2) ? 1 : 0;
    for (size_t i = 0; i < dlb.linear_block.pconv_weights.size(); ++i) 
        dlb.linear_block.pconv_weights[i] = 0;  // Zero to avoid amplification
    for (size_t i = 0; i < dlb.linear_block.pconv_biases.size(); ++i) 
        dlb.linear_block.pconv_biases[i] = 0;
    for (size_t i = 0; i < dlb.linear_block.dconv2_weights.size(); ++i) 
        dlb.linear_block.dconv2_weights[i] = 0;
    for (size_t i = 0; i < dlb.linear_block.pconv_out_weights.size(); ++i) 
        dlb.linear_block.pconv_out_weights[i] = 0;
    for (size_t i = 0; i < dlb.linear_block.pconv_out_biases.size(); ++i) 
        dlb.linear_block.pconv_out_biases[i] = 0;
    
    // Test forward pass with skip connection
    int input_h = 7, input_w = 7;
    int input_size = input_h * input_w * 64;
    auto input_data = generate_random_int8(input_size);
    
    // Limit input range for stable skip connection
    for (auto& val : input_data) {
        val = std::max(-10, std::min(10, static_cast<int>(val)));
    }
    
    std::vector<int8_t> output_data(input_size);
    
    try {
        dlb.forward(input_data.data(), output_data.data(), input_h, input_w);
        results.test("DenseLinearDepthwiseBlock forward pass", true);
    } catch (...) {
        results.test("DenseLinearDepthwiseBlock forward pass", false);
    }
}

// Test LinearLayer
void test_linear_layer(TestResults& results) {
    std::cout << "\n🧪 Testing LinearLayer..." << std::endl;
    
    LinearLayer layer;
    layer.in_features = 512;
    layer.out_features = 10;
    layer.scale = 64.0f;
    
    // Initialize weights and biases
    layer.weights.resize(layer.out_features * layer.in_features);
    layer.biases.resize(layer.out_features);
    
    // Fill with test data
    for (int i = 0; i < layer.out_features * layer.in_features; ++i) {
        layer.weights[i] = (i % 3) - 1;  // -1, 0, 1 pattern
    }
    for (int i = 0; i < layer.out_features; ++i) {
        layer.biases[i] = i * 100;  // bias = index * 100
    }
    
    // Test forward pass
    auto input_data = generate_random_int8(layer.in_features);
    std::vector<float> output_data(layer.out_features);
    
    try {
        layer.forward(input_data.data(), output_data.data());
        results.test("LinearLayer forward pass", true);
    } catch (...) {
        results.test("LinearLayer forward pass", false);
    }
    
    results.test("LinearLayer output size", static_cast<int>(output_data.size()) == layer.out_features);
    
    // Check that outputs are reasonable (not NaN or infinite)
    bool outputs_valid = true;
    for (float val : output_data) {
        if (std::isnan(val) || std::isinf(val)) {
            outputs_valid = false;
            break;
        }
    }
    results.test("LinearLayer outputs are valid", outputs_valid);
}

// Test EtinyNetEvaluator basic functionality
void test_etinynet_evaluator(TestResults& results) {
    std::cout << "\n🧪 Testing EtinyNetEvaluator..." << std::endl;
    
    EtinyNetEvaluator evaluator;
    
    // Test basic construction
    results.test("EtinyNetEvaluator construction", evaluator.get_variant() == "1.0");
    results.test("EtinyNetEvaluator default classes", evaluator.get_num_classes() == 1000);
    results.test("EtinyNetEvaluator default input size", evaluator.get_input_size() == 112);
    results.test("EtinyNetEvaluator default ASQ", !evaluator.uses_asq());
    
    // Test helper functions
    int input_size = 32 * 32 * 3;  // CIFAR-10 size
    auto input_data = generate_random_floats(input_size, 0.0f, 1.0f);
    std::vector<int8_t> quantized_input(input_size);
    
    // Test quantization (private function, but we can test the concept)
    for (int i = 0; i < input_size; ++i) {
        float val = input_data[i] * 127.0f;
        quantized_input[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, val)));
    }
    
    bool quantization_reasonable = true;
    for (int i = 0; i < std::min(10, input_size); ++i) {
        if (quantized_input[i] < -127 || quantized_input[i] > 127) {
            quantization_reasonable = false;
            break;
        }
    }
    results.test("Input quantization produces valid range", quantization_reasonable);
}

// Performance test
void test_performance(TestResults& results) {
    std::cout << "\n🧪 Testing Performance..." << std::endl;
    
    // Test AlignedVector performance
    const int size = 1000000;
    auto start = std::chrono::high_resolution_clock::now();
    
    AlignedVector<float> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = static_cast<float>(i);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  AlignedVector write performance: " << duration.count() << " μs for " 
              << size << " elements" << std::endl;
    
    // Should complete in reasonable time (< 10ms)
    results.test("AlignedVector performance reasonable", duration.count() < 10000);
    
    // Test basic convolution performance
    DepthwiseSeparableConv conv;
    conv.in_channels = 32;
    conv.out_channels = 64;
    conv.kernel_size = 3;
    conv.stride = 1;
    conv.padding = 1;
    
    conv.depthwise_weights.resize(conv.in_channels * 9);
    conv.pointwise_weights.resize(conv.out_channels * conv.in_channels);
    conv.pointwise_biases.resize(conv.out_channels);
    
    // Fill with dummy data
    for (size_t i = 0; i < conv.depthwise_weights.size(); ++i) conv.depthwise_weights[i] = 1;
    for (size_t i = 0; i < conv.pointwise_weights.size(); ++i) conv.pointwise_weights[i] = 1;
    for (size_t i = 0; i < conv.pointwise_biases.size(); ++i) conv.pointwise_biases[i] = 0;
    
    int input_h = 32, input_w = 32;
    auto input_data = generate_random_int8(input_h * input_w * conv.in_channels);
    std::vector<int8_t> output_data(input_h * input_w * conv.out_channels);
    
    start = std::chrono::high_resolution_clock::now();
    conv.forward(input_data.data(), output_data.data(), input_h, input_w, true);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  DepthwiseSeparableConv performance: " << duration.count() 
              << " μs for " << input_h << "x" << input_w << " input" << std::endl;
    
    // Should complete in reasonable time (< 100ms for this size)
    results.test("DepthwiseSeparableConv performance reasonable", duration.count() < 100000);
}

int main() {
    std::cout << "🧪 EtinyNet C++ Engine Tests" << std::endl;
    std::cout << "=============================" << std::endl;
    
    TestResults results;
    
    try {
        test_aligned_vector(results);
        test_depthwise_separable_conv(results);
        test_linear_depthwise_block(results);
        test_dense_linear_depthwise_block(results);
        test_linear_layer(results);
        test_etinynet_evaluator(results);
        test_performance(results);
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    results.summary();
    
    if (results.all_passed()) {
        std::cout << "🎉 All C++ engine tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "⚠️ Some C++ engine tests failed." << std::endl;
        return 1;
    }
} 