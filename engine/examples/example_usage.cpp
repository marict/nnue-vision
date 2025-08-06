#include "../include/nnue_engine.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace nnue;

// Define constants for this example (using the defaults from the header)
constexpr int INPUT_IMAGE_SIZE = DEFAULT_INPUT_IMAGE_SIZE;
constexpr int INPUT_CHANNELS = DEFAULT_INPUT_CHANNELS;
constexpr int OUTPUT_CHANNELS = DEFAULT_OUTPUT_CHANNELS;
constexpr int CONV_KERNEL_SIZE = DEFAULT_CONV_KERNEL_SIZE;
constexpr int GRID_FEATURES = DEFAULT_OUTPUT_GRID_SIZE * DEFAULT_OUTPUT_GRID_SIZE * DEFAULT_OUTPUT_CHANNELS;
constexpr int L1_SIZE = DEFAULT_L1_SIZE;
constexpr int L2_SIZE = DEFAULT_L2_SIZE;
constexpr int L3_SIZE = DEFAULT_L3_SIZE;

// Helper function to load an image from RGB data
std::vector<float> load_test_image() {
    // Create a test image with some pattern
    // In practice, this would come from actual image data
    std::vector<float> image(INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * INPUT_CHANNELS);
    
    // Create a simple gradient pattern
    for (int h = 0; h < INPUT_IMAGE_SIZE; ++h) {
        for (int w = 0; w < INPUT_IMAGE_SIZE; ++w) {
            for (int c = 0; c < INPUT_CHANNELS; ++c) {
                int idx = (h * INPUT_IMAGE_SIZE + w) * INPUT_CHANNELS + c;
                
                // Simple pattern: varies with position and channel
                float value = (static_cast<float>(h + w + c * 32) / (INPUT_IMAGE_SIZE * 2 + 3 * 32));
                image[idx] = std::max(0.0f, std::min(1.0f, value));
            }
        }
    }
    
    return image;
}

// Generate a batch of random test images
std::vector<std::vector<float>> generate_test_batch(int batch_size, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<std::vector<float>> batch;
    
    for (int b = 0; b < batch_size; ++b) {
        std::vector<float> image(INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * INPUT_CHANNELS);
        
        for (auto& pixel : image) {
            pixel = dis(gen);
        }
        
        batch.push_back(std::move(image));
    }
    
    return batch;
}

// Benchmark the evaluation speed
void benchmark_evaluation(NNUEEvaluator& evaluator, int num_evaluations = 1000) {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
    // Generate test images
    auto test_images = generate_test_batch(100);
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        evaluator.evaluate(test_images[i % test_images.size()].data());
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    float total_score = 0.0f;
    for (int i = 0; i < num_evaluations; ++i) {
        float score = evaluator.evaluate(test_images[i % test_images.size()].data());
        total_score += score;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Evaluated " << num_evaluations << " images in " 
              << duration.count() << " microseconds" << std::endl;
    std::cout << "Average time per evaluation: " 
              << static_cast<double>(duration.count()) / num_evaluations << " microseconds" << std::endl;
    std::cout << "Evaluations per second: " 
              << static_cast<double>(num_evaluations) / duration.count() * 1000000 << std::endl;
    std::cout << "Average score: " << total_score / num_evaluations << std::endl;
}

// Demonstrate different layer stack usage
void test_layer_stacks(NNUEEvaluator& evaluator) {
    std::cout << "\n=== Layer Stack Testing ===" << std::endl;
    
    auto test_image = load_test_image();
    int num_stacks = evaluator.get_num_layer_stacks();
    
    std::cout << "Available layer stacks: " << num_stacks << std::endl;
    
    for (int i = 0; i < num_stacks; ++i) {
        float score = evaluator.evaluate(test_image.data(), i);
        std::cout << "Layer stack " << i << ": " << score << std::endl;
    }
}

// Demonstrate batch processing
void batch_processing_example(NNUEEvaluator& evaluator) {
    std::cout << "\n=== Batch Processing Example ===" << std::endl;
    
    const int batch_size = 32;
    auto batch = generate_test_batch(batch_size);
    
    std::vector<float> scores;
    scores.reserve(batch_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& image : batch) {
        float score = evaluator.evaluate(image.data());
        scores.push_back(score);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate statistics
    float min_score = *std::min_element(scores.begin(), scores.end());
    float max_score = *std::max_element(scores.begin(), scores.end());
    float avg_score = std::accumulate(scores.begin(), scores.end(), 0.0f) / batch_size;
    
    std::cout << "Processed batch of " << batch_size << " images" << std::endl;
    std::cout << "Total time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Average time per image: " << duration.count() / batch_size << " microseconds" << std::endl;
    std::cout << "Score statistics:" << std::endl;
    std::cout << "  Min: " << min_score << std::endl;
    std::cout << "  Max: " << max_score << std::endl;
    std::cout << "  Average: " << avg_score << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== NNUE Engine Example Usage ===" << std::endl;
    
    // Create evaluator
    NNUEEvaluator evaluator;
    
    // Check for model file argument
    std::string model_path = "model.nnue";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Attempting to load model from: " << model_path << std::endl;
    
    // Try to load the model
    if (!evaluator.load_model(model_path)) {
        std::cout << "Failed to load model. Running demonstration with dummy operations..." << std::endl;
        std::cout << "\nTo use this example with a real model:" << std::endl;
        std::cout << "1. Train a model using the Python scripts" << std::endl;
        std::cout << "2. Serialize it to .nnue format using serialize.py" << std::endl;
        std::cout << "3. Run: " << argv[0] << " path/to/your/model.nnue" << std::endl;
        
        // Run basic component tests instead
        std::cout << "\n=== Running Component Tests ===" << std::endl;
        
        // Test individual components
        std::cout << "Testing ConvLayer..." << std::endl;
        ConvLayer conv;
        conv.scale = 64.0f;
        conv.weights.resize(OUTPUT_CHANNELS * INPUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE);
        conv.biases.resize(OUTPUT_CHANNELS);
        std::fill(conv.weights.data(), conv.weights.data() + conv.weights.size(), 1);
        std::fill(conv.biases.data(), conv.biases.data() + conv.biases.size(), 0);
        
        auto test_image = load_test_image();
        AlignedVector<int8_t> conv_output(GRID_FEATURES);
        conv.forward(test_image.data(), conv_output.data(), INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1);
        std::cout << "ConvLayer test completed." << std::endl;
        
        return 0;
    }
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Visual threshold: " << evaluator.get_visual_threshold() << std::endl;
    
    // Basic evaluation example
    std::cout << "\n=== Basic Evaluation Example ===" << std::endl;
    auto test_image = load_test_image();
    float score = evaluator.evaluate(test_image.data());
    std::cout << "Test image evaluation score: " << score << std::endl;
    
    // Test different layer stacks
    test_layer_stacks(evaluator);
    
    // Batch processing demonstration
    batch_processing_example(evaluator);
    
    // Performance benchmark
    benchmark_evaluation(evaluator);
    
    // Show memory usage information
    std::cout << "\n=== Memory Usage Information ===" << std::endl;
    std::cout << "Architecture constants:" << std::endl;
    std::cout << "  Input image size: " << INPUT_IMAGE_SIZE << "x" << INPUT_IMAGE_SIZE << "x" << INPUT_CHANNELS << std::endl;
    std::cout << "  Grid features: " << GRID_FEATURES << std::endl;
    std::cout << "  L1 size: " << L1_SIZE << std::endl;
    std::cout << "  L2 size: " << L2_SIZE << std::endl;
    std::cout << "  L3 size: " << L3_SIZE << std::endl;
    
    // Estimate memory usage
    size_t conv_weights = OUTPUT_CHANNELS * INPUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;
    size_t ft_weights = GRID_FEATURES * L1_SIZE * sizeof(int16_t);
    size_t l1_weights = L2_SIZE * L1_SIZE;
    size_t l2_weights = L3_SIZE * L2_SIZE;
    size_t output_weights = 1 * L3_SIZE;
    
    size_t total_model_size = conv_weights + ft_weights + l1_weights + l2_weights + output_weights;
    
    std::cout << "Estimated model size: " << total_model_size / 1024 << " KB" << std::endl;
    
    std::cout << "\nExample completed successfully!" << std::endl;
    return 0;
} 