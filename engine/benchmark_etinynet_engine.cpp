// engine/benchmark_etinynet_engine.cpp
#include "include/nnue_engine.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <numeric>
#include <algorithm>

using namespace nnue;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.etiny>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    std::cout << "🚀 EtinyNet C++ Engine Performance Benchmark\n";
    std::cout << "===========================================\n\n";

    // Load EtinyNet model
    EtinyNetEvaluator evaluator;
    if (!evaluator.load_model(model_path)) {
        std::cerr << "❌ Failed to load model: " << model_path << std::endl;
        return 1;
    }

    const int input_size = evaluator.get_input_size();
    const int num_classes = evaluator.get_num_classes();

    std::cout << "✅ Model loaded successfully\n";
    std::cout << "   • Variant: EtinyNet-" << evaluator.get_variant() << "\n";
    std::cout << "   • Input size: " << input_size << "x" << input_size << "\n";
    std::cout << "   • Classes: " << num_classes << "\n\n";

    // Prepare random input image generator
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const size_t image_elements = static_cast<size_t>(input_size) * input_size * 3;
    std::vector<float> image(image_elements);
    std::vector<float> output(num_classes);

    auto fill_random_image = [&]() {
        for (auto& px : image) {
            px = dist(rng);
        }
    };

    // Warm-up
    for (int i = 0; i < 10; ++i) {
        fill_random_image();
        evaluator.evaluate(image.data(), output.data(), input_size, input_size);
    }

    const int iterations = 1000;
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        fill_random_image();
        const auto start = high_resolution_clock::now();
        evaluator.evaluate(image.data(), output.data(), input_size, input_size);
        const auto end = high_resolution_clock::now();
        const double ms = duration_cast<nanoseconds>(end - start).count() / 1'000'000.0;
        times.push_back(ms);
    }

    const double sum = std::accumulate(times.begin(), times.end(), 0.0);
    const double avg = sum / times.size();
    const auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());

    std::cout << std::left << std::setw(20) << "Scenario"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)" << "\n";
    std::cout << std::string(60, '-') << "\n";

    std::cout << std::left << std::setw(20) << "RandomImage"
              << std::setw(12) << std::fixed << std::setprecision(4) << avg
              << std::setw(12) << std::fixed << std::setprecision(4) << *min_it
              << std::setw(12) << std::fixed << std::setprecision(4) << *max_it << "\n";

    // Extra machine-readable line for easier parsing by Python script
    std::cout << "RESULT_AVG_MS: " << avg << "\n";

    return 0;
} 