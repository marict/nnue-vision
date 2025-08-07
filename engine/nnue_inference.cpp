#include "include/nnue_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

// Simple CLI to run NNUEEvaluator on a single image stored as raw float32 RGB array.
// Usage: nnue_inference <model.nnue> <image.bin> <H> <W>
// Writes single evaluation result to stdout: <value>

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <model.nnue> <image.bin> <H> <W>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    int H = std::atoi(argv[3]);
    int W = std::atoi(argv[4]);

    // Load image
    const size_t elem_count = static_cast<size_t>(H) * W * 3;
    std::vector<float> image(elem_count);
    std::ifstream img_file(image_path, std::ios::binary);
    if (!img_file.is_open()) {
        std::cerr << "Cannot open image file: " << image_path << std::endl;
        return 1;
    }
    img_file.read(reinterpret_cast<char*>(image.data()), elem_count * sizeof(float));
    if (!img_file) {
        std::cerr << "Failed to read image data" << std::endl;
        return 1;
    }

    // Load model
    nnue::NNUEEvaluator evaluator;
    if (!evaluator.load_model(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Evaluate (use layer stack 0 for inference)
    float result = evaluator.evaluate(image.data(), H, W, 0);

    // Get active features for density calculation
    std::vector<int> active_features;
    evaluator.get_active_features(active_features);
    
    // Calculate density (non-zero ratio)
    int total_features = evaluator.get_total_features();
    float density = total_features > 0 ? static_cast<float>(active_features.size()) / total_features : 0.0f;

    // Print result and density (format: result,density)
    std::cout << std::fixed << std::setprecision(10) << result << "," << density << std::endl;

    return 0;
}
