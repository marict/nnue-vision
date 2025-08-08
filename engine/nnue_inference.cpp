#include "include/nnue_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

// CLI: nnue_inference <model.nnue> <image.bin> <H> <W>
// Output: CSV "logit_0,...,logit_{C-1},density"

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

    // Evaluate logits via multiclass path if available; otherwise single-score
    std::vector<float> logits = evaluator.evaluate_logits(image.data(), H, W, 0);
    if (logits.empty()) {
        float single = evaluator.evaluate(image.data(), H, W, 0);
        logits.push_back(single);
    }

    // Get active features for density calculation AFTER logits path (evaluate also sets them)
    std::vector<int> active_features;
    evaluator.get_active_features(active_features);
    int total_features = evaluator.get_total_features();
    float density = total_features > 0 ? static_cast<float>(active_features.size()) / total_features : 0.0f;

    // Emit logits as CSV followed by density
    std::cout << std::fixed << std::setprecision(10);
    for (size_t i = 0; i < logits.size(); ++i) {
        std::cout << logits[i];
        std::cout << ",";
    }
    std::cout << density << std::endl;

    return 0;
}
