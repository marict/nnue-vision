#include "include/nnue_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

// Simple CLI to run NNUEEvaluator on a single image stored as raw float32 RGB array.
// Usage: nnue_inference <model.nnue> <image.bin> <H> <W>
// Writes logits and density in one line as CSV: v0,v1,...,v{C-1},density

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

    // Evaluate (use layer stack 0). For multi-class, we approximate by sampling layer stacks if available
    // and constructing a logits vector of size C using a simple linear head assumption.
    // If only a single output head exists, we output one value as class-0 logit and zeros for others.

    // Determine number of classes from serialized metadata: use l3_size as a proxy for number of classes if available
    // The current NNUE serialization now stores output layer with out_out_size=num_classes.
    // We cannot query it directly here, so we derive it from the LayerStack's l3_size via a helper getter pattern.
    // Fallback to 1 if unknown.

    // Build multi-class logits if available; fallback to single-score vector
    std::vector<float> logits = evaluator.evaluate_logits(image.data(), H, W, 0);
    bool have_multiclass = !logits.empty();
    float single = have_multiclass ? logits[0] : evaluator.evaluate(image.data(), H, W, 0);

    // Get active features for density calculation
    std::vector<int> active_features;
    evaluator.get_active_features(active_features);
    
    // Calculate density (non-zero ratio)
    int total_features = evaluator.get_total_features();
    float density = total_features > 0 ? static_cast<float>(active_features.size()) / total_features : 0.0f;

    // Emit logits as CSV followed by density. If only single present, expand to CIFAR-10-compatible vector
    const int num_classes = have_multiclass ? static_cast<int>(logits.size()) : 10;
    std::cout << std::fixed << std::setprecision(10);
    for (int i = 0; i < num_classes; ++i) {
        float v = have_multiclass ? logits[static_cast<size_t>(i)] : ((i == 0) ? single : 0.0f);
        std::cout << v << (i + 1 < num_classes ? "," : ",");
    }
    std::cout << density << std::endl;

    return 0;
}
