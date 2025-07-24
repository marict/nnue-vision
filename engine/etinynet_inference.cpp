#include "include/nnue_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

// Simple CLI to run EtinyNetEvaluator on a single image stored as raw float32 RGB array.
// Usage: etinynet_inference <model.etiny> <image.bin> <H> <W>
// Writes logits to stdout lines: RESULT_<idx>: <value>

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <model.etiny> <image.bin> <H> <W>" << std::endl;
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
    nnue::EtinyNetEvaluator evaluator;
    if (!evaluator.load_model(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    int num_classes = evaluator.get_num_classes();
    std::vector<float> output(num_classes, 0.0f);

    // Evaluate
    evaluator.evaluate(image.data(), output.data(), H, W);

    // Print results
    std::cout << std::fixed << std::setprecision(10);
    for (int i = 0; i < num_classes; ++i) {
        std::cout << "RESULT_" << i << ": " << output[i] << std::endl;
    }

    return 0;
} 