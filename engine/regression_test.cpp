#include "include/nnue_engine.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip> // Required for std::fixed and std::setprecision
#include <chrono>  // Required for performance benchmarking

using namespace nnue;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.nnue> [feature1 feature2 ...]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    std::cout << "=== NNUE Regression Test ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    
    // Load the model
    NNUEEvaluator evaluator;
    if (!evaluator.load_model(model_path)) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }
    
    std::cout << "Features: " << evaluator.get_num_features() << std::endl;
    std::cout << "L1 size: " << evaluator.get_l1_size() << std::endl;
    std::cout << "Grid size: " << evaluator.get_grid_size() << std::endl;
    std::cout << "Channels per square: " << evaluator.get_num_channels_per_square() << std::endl;
    std::cout << "Visual threshold: " << evaluator.get_visual_threshold() << std::endl;
    std::cout << "Layer stacks: " << evaluator.get_num_layer_stacks() << std::endl;
    
    // Parse test features from command line
    std::vector<int> test_features;
    for (int i = 2; i < argc; ++i) {
        int feature = std::atoi(argv[i]);
        if (feature >= 0 && feature < evaluator.get_num_features()) {
            test_features.push_back(feature);
        }
    }
    
    std::cout << "Test features (" << test_features.size() << "):";
    for (int f : test_features) {
        std::cout << " " << f;
    }
    std::cout << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Incremental Evaluation Test ===" << std::endl;
    
    // Test incremental evaluation for each layer stack
    for (int ls = 0; ls < evaluator.get_num_layer_stacks(); ++ls) {
        evaluator.mark_dirty(); // Force fresh start
        float result = evaluator.evaluate_incremental(test_features, ls);
        std::cout << "RESULT_INCREMENTAL_" << ls << ": " << std::fixed << std::setprecision(10) << result << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Image Evaluation Test ===" << std::endl;
    
    // Test image evaluation (using a simple test image)
    int grid_size = evaluator.get_grid_size();
    int image_size = grid_size * 12; // Make image larger than grid
    std::vector<float> test_image(image_size * image_size * 3, 0.1f); // Small positive values
    
    for (int ls = 0; ls < evaluator.get_num_layer_stacks(); ++ls) {
        float result = evaluator.evaluate(test_image.data(), image_size, image_size, ls);
        std::cout << "RESULT_IMAGE_" << ls << ": " << std::fixed << std::setprecision(10) << result << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Numerical Stability Test ===" << std::endl;
    
    // Test with empty features
    std::vector<int> empty_features;
    evaluator.mark_dirty();
    float empty_result = evaluator.evaluate_incremental(empty_features, 0);
    std::cout << "RESULT_EMPTY: " << std::fixed << std::setprecision(10) << empty_result << std::endl;
    
    // Test with single feature
    std::vector<int> single_feature = {0};
    evaluator.mark_dirty();
    float single_result = evaluator.evaluate_incremental(single_feature, 0);
    std::cout << "RESULT_SINGLE: " << std::fixed << std::setprecision(10) << single_result << std::endl;
    
    // Test with repeated features
    std::vector<int> repeated_features = {10, 20, 30, 40, 50};
    evaluator.mark_dirty();
    float repeated_result = evaluator.evaluate_incremental(repeated_features, 0);
    std::cout << "RESULT_REPEATED: " << std::fixed << std::setprecision(10) << repeated_result << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Performance Benchmark ===" << std::endl;
    
    // Quick performance test
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        evaluator.evaluate_incremental(test_features, 0);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Total time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Average time: " << static_cast<double>(duration.count()) / iterations << " microseconds per evaluation" << std::endl;
    std::cout << "Throughput: " << static_cast<double>(iterations) * 1000000.0 / duration.count() << " evaluations per second" << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Test Complete ===" << std::endl;
    
    return 0;
} 