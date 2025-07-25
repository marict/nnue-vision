#include "include/nnue_engine.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <numeric> // Required for std::iota and std::accumulate
#include <algorithm> // Required for std::shuffle

using namespace nnue;
using namespace std::chrono;

struct BenchmarkResult {
    std::string scenario;
    int num_features;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    std::vector<int> active_features;
};

std::vector<int> generate_sparse_features(int total_features, double sparsity_ratio, std::mt19937& rng) {
    int num_active = std::max(1, static_cast<int>(total_features * sparsity_ratio));
    
    std::vector<int> all_features(total_features);
    std::iota(all_features.begin(), all_features.end(), 0);
    
    std::shuffle(all_features.begin(), all_features.end(), rng);
    
    std::vector<int> active_features(all_features.begin(), all_features.begin() + num_active);
    std::sort(active_features.begin(), active_features.end());
    
    return active_features;
}

std::vector<int> generate_varied_features(const std::vector<int>& base_features, 
                                         int total_features, double sparsity_ratio, 
                                         double overlap_ratio, std::mt19937& rng) {
    int target_num_features = std::max(1, static_cast<int>(total_features * sparsity_ratio));
    int keep_count = static_cast<int>(base_features.size() * overlap_ratio);
    
    std::vector<int> result;
    result.reserve(target_num_features);
    
    // Keep some features from base set
    std::vector<int> base_copy = base_features;
    std::shuffle(base_copy.begin(), base_copy.end(), rng);
    for (int i = 0; i < keep_count && i < static_cast<int>(base_copy.size()); ++i) {
        result.push_back(base_copy[i]);
    }
    
    // Add new random features
    std::vector<int> all_features(total_features);
    std::iota(all_features.begin(), all_features.end(), 0);
    std::shuffle(all_features.begin(), all_features.end(), rng);
    
    for (int feature : all_features) {
        if (std::find(result.begin(), result.end(), feature) == result.end()) {
            result.push_back(feature);
            if (static_cast<int>(result.size()) >= target_num_features) break;
        }
    }
    
    std::sort(result.begin(), result.end());
    return result;
}

BenchmarkResult benchmark_scenario_with_variation(NNUEEvaluator& evaluator, const std::string& scenario_name,
                                                 double sparsity_ratio, int iterations = 1000) {
    std::vector<double> times;
    times.reserve(iterations);
    
    std::mt19937 rng(42); // Fixed seed for reproducibility
    
    // Generate initial feature set
    std::vector<int> current_features = generate_sparse_features(
        evaluator.get_num_features(), sparsity_ratio, rng);
    
    // Warmup with varying features
    for (int i = 0; i < 10; ++i) {
        // Generate slightly different feature set for each warmup
        std::vector<int> warmup_features = generate_sparse_features(
            evaluator.get_num_features(), sparsity_ratio, rng);
        evaluator.evaluate_incremental(warmup_features, 0);
    }
    
    // Benchmark with fluctuating features to trigger incremental updates
    for (int i = 0; i < iterations; ++i) {
        // Generate new feature set with some overlap to simulate realistic position changes
        std::vector<int> new_features = generate_varied_features(
            current_features, evaluator.get_num_features(), sparsity_ratio, 0.8, rng);
        
        auto start = high_resolution_clock::now();
        
        evaluator.evaluate_incremental(new_features, 0);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);
        times.push_back(duration.count() / 1000000.0); // Convert to milliseconds
        
        // Update current features for next iteration
        current_features = new_features;
    }
    
    // Calculate statistics
    double sum = 0.0;
    double min_time = times[0];
    double max_time = times[0];
    
    for (double time : times) {
        sum += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    
    return {
        scenario_name,
        static_cast<int>(current_features.size()),
        sum / times.size(),
        min_time,
        max_time,
        current_features
    };
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.nnue>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    std::cout << "🚀 NNUE C++ Engine Performance Benchmark\n";
    std::cout << "=========================================\n\n";
    
    // Load the model
    NNUEEvaluator evaluator;
    if (!evaluator.load_model(model_path)) {
        std::cerr << "❌ Failed to load model: " << model_path << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model loaded successfully\n";
    std::cout << "   • Features: " << evaluator.get_num_features() << "\n";
    std::cout << "   • Architecture: " << evaluator.get_num_features() 
              << " → " << evaluator.get_l1_size()
              << " → " << evaluator.get_l2_size() 
              << " → " << evaluator.get_l3_size() << " → 1\n";
    std::cout << "   • Layer stacks: " << evaluator.get_num_layer_stacks() << "\n\n";
    
    // Enable incremental updates for maximum performance
    evaluator.enable_incremental(true);
    
    // Define test scenarios
    struct Scenario {
        std::string name;
        double sparsity_ratio;
        std::string description;
    };
    
    std::vector<Scenario> scenarios = {
        {"Chess-like (0.1%)", 0.001, "Ultra sparse like chess engines"},
        {"Very Sparse (1%)", 0.01, "Highly sparse"},
        {"Sparse (5%)", 0.05, "Moderately sparse"},
        {"Medium (25%)", 0.25, "Medium sparsity"},
        {"Dense (90%)", 0.90, "Mostly dense"}
    };
    
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::vector<BenchmarkResult> results;
    
    std::cout << "🔬 Benchmarking Performance:\n";
    std::cout << std::left << std::setw(20) << "Scenario" 
              << std::setw(12) << "Features"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)" 
              << std::setw(12) << "Max (ms)"
              << "Description\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (const auto& scenario : scenarios) {
        // Generate sparse feature set
        std::vector<int> active_features = generate_sparse_features(
            evaluator.get_num_features(), scenario.sparsity_ratio, rng);
        
        // Reset incremental state for fair comparison
        evaluator.mark_dirty();
        
        // Benchmark this scenario
        BenchmarkResult result = benchmark_scenario_with_variation(evaluator, scenario.name, scenario.sparsity_ratio, 1000);
        results.push_back(result);
        
        std::cout << std::left << std::setw(20) << result.scenario
                  << std::setw(12) << result.num_features
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.avg_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.min_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.max_time_ms
                  << scenario.description << "\n";
    }
    
    std::cout << "\n⚡ Speedup Analysis:\n";
    std::cout << std::string(60, '-') << "\n";
    
    // Use dense case as baseline
    double baseline_time = results.back().avg_time_ms;
    
    std::cout << std::left << std::setw(20) << "Scenario"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Speedup"
              << "Efficiency\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& result : results) {
        double speedup = baseline_time / result.avg_time_ms;
        double efficiency = (result.num_features == 0) ? 0.0 : 
            static_cast<double>(evaluator.get_num_features()) / result.num_features;
        
        std::cout << std::left << std::setw(20) << result.scenario
                  << std::setw(15) << std::fixed << std::setprecision(4) << result.avg_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << speedup << "x"
                  << std::fixed << std::setprecision(1) << efficiency << "x theoretical\n";
    }
    
    // Test incremental updates specifically
    std::cout << "\n🔄 Incremental Update Performance:\n";
    std::cout << std::string(50, '-') << "\n";
    
    // Start with sparse features
    std::vector<int> initial_features = generate_sparse_features(evaluator.get_num_features(), 0.01, rng);
    evaluator.refresh_accumulator(initial_features);
    
    // Measure incremental vs full refresh
    const int incremental_iterations = 100;
    
    // Incremental update: change 10% of features
    std::vector<double> incremental_times;
    for (int i = 0; i < incremental_iterations; ++i) {
        // Generate slightly different feature set
        std::vector<int> new_features = generate_sparse_features(evaluator.get_num_features(), 0.01, rng);
        
        auto start = high_resolution_clock::now();
        evaluator.evaluate_incremental(new_features, 0);
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<nanoseconds>(end - start);
        incremental_times.push_back(duration.count() / 1000000.0);
    }
    
    // Full refresh: reset and recompute
    std::vector<double> full_refresh_times;
    for (int i = 0; i < incremental_iterations; ++i) {
        std::vector<int> new_features = generate_sparse_features(evaluator.get_num_features(), 0.01, rng);
        
        evaluator.mark_dirty(); // Force full refresh
        
        auto start = high_resolution_clock::now();
        evaluator.evaluate_incremental(new_features, 0);
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<nanoseconds>(end - start);
        full_refresh_times.push_back(duration.count() / 1000000.0);
    }
    
    double avg_incremental = std::accumulate(incremental_times.begin(), incremental_times.end(), 0.0) / incremental_times.size();
    double avg_full_refresh = std::accumulate(full_refresh_times.begin(), full_refresh_times.end(), 0.0) / full_refresh_times.size();
    double incremental_speedup = avg_full_refresh / avg_incremental;
    
    std::cout << "Full Refresh:     " << std::fixed << std::setprecision(4) << avg_full_refresh << " ms\n";
    std::cout << "Incremental:      " << std::fixed << std::setprecision(4) << avg_incremental << " ms\n";
    std::cout << "Incremental Speedup: " << std::fixed << std::setprecision(1) << incremental_speedup << "x\n";
    
    std::cout << "\n🏆 C++ Engine Performance Summary:\n";
    std::cout << "==================================\n";
    
    auto chess_result = results[0]; // Chess-like scenario
    auto dense_result = results.back(); // Dense scenario
    double chess_speedup = dense_result.avg_time_ms / chess_result.avg_time_ms;
    
    std::cout << "Best case (chess-like): " << std::fixed << std::setprecision(4) << chess_result.avg_time_ms << " ms\n";
    std::cout << "Worst case (dense):     " << std::fixed << std::setprecision(4) << dense_result.avg_time_ms << " ms\n";
    std::cout << "Sparsity advantage:     " << std::fixed << std::setprecision(1) << chess_speedup << "x\n";
    
    // Output parseable results for Python script
    std::cout << "\n📊 DENSITY_RESULTS:\n";
    for (const auto& result : results) {
        std::cout << "DENSITY_RESULT:" << result.scenario << ":" << std::fixed << std::setprecision(4) << result.avg_time_ms << "\n";
    }
    
    // Overall average for fallback parsing
    double overall_avg = 0.0;
    for (const auto& result : results) {
        overall_avg += result.avg_time_ms;
    }
    overall_avg /= results.size();
    std::cout << "RESULT_AVG_MS: " << std::fixed << std::setprecision(4) << overall_avg << "\n";
    
    return 0;
} 