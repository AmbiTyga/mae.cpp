#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <deque>
#include <map>
#include <algorithm>
#include <limits>

using json = nlohmann::json;

// Configuration structure
struct PretrainConfig {
    // Model config
    std::string model_type;
    bool norm_pix_loss;
    float mask_ratio;
    
    // Optimization config
    int64_t batch_size;
    double base_lr;
    double min_lr;
    double weight_decay;
    std::string optimizer_type;
    double adam_beta1;
    double adam_beta2;
    double gradient_clip;
    
    // Schedule config
    int64_t epochs;
    int64_t warmup_epochs;
    int64_t start_epoch;
    
    // Data config
    std::string data_path;
    int64_t input_size;
    int64_t num_workers;
    
    // Augmentation config
    std::array<float, 2> crop_scale;
    std::string interpolation;
    bool random_flip;
    std::array<float, 3> normalize_mean;
    std::array<float, 3> normalize_std;
    
    // Checkpointing config
    std::string checkpoint_dir;
    int64_t save_freq_epochs;
    int64_t save_freq_steps;
    int64_t keep_last_n;
    
    // Logging config
    int64_t print_freq;
    int64_t console_clear_freq;
    
    // Misc config
    int64_t seed;
    std::string resume;
    bool auto_resume;
    
    // Device
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    
    // Load from JSON file
    static PretrainConfig from_json(const std::string& config_path);
};

// Learning rate scheduler
class CosineAnnealingWarmupScheduler {
public:
    CosineAnnealingWarmupScheduler(torch::optim::Optimizer& optimizer,
                                  int64_t warmup_epochs,
                                  int64_t total_epochs,
                                  double base_lr,
                                  double min_lr = 0.0)
        : optimizer_(optimizer),
          warmup_epochs_(warmup_epochs),
          total_epochs_(total_epochs),
          base_lr_(base_lr),
          min_lr_(min_lr) {}
    
    double get_lr(int64_t epoch) const {
        if (epoch < warmup_epochs_) {
            // Linear warmup
            return base_lr_ * epoch / warmup_epochs_;
        } else {
            // Cosine annealing
            double progress = static_cast<double>(epoch - warmup_epochs_) / 
                            (total_epochs_ - warmup_epochs_);
            return min_lr_ + (base_lr_ - min_lr_) * 0.5 * 
                   (1.0 + std::cos(M_PI * progress));
        }
    }
    
    void step(int64_t epoch) {
        double lr = get_lr(epoch);
        for (auto& param_group : optimizer_.param_groups()) {
            param_group.options().set_lr(lr);
        }
    }
    
private:
    torch::optim::Optimizer& optimizer_;
    int64_t warmup_epochs_;
    int64_t total_epochs_;
    double base_lr_;
    double min_lr_;
};

// Note: RandomResizedCrop is already defined in data_loader.hpp
// We'll use that implementation

// Metric logger for training statistics
class MetricLogger {
public:
    MetricLogger(const std::string& delimiter = "  ") : delimiter_(delimiter) {}
    
    void update(const std::string& name, double value, int n = 1) {
        if (meters_.find(name) == meters_.end()) {
            meters_[name] = SmoothedValue();
        }
        meters_[name].update(value, n);
    }
    
    double get_avg(const std::string& name) const {
        auto it = meters_.find(name);
        if (it != meters_.end()) {
            return it->second.avg;
        }
        return 0.0;
    }
    
    std::string get_str(const std::string& name) const {
        auto it = meters_.find(name);
        if (it != meters_.end()) {
            return it->second.to_string();
        }
        return "";
    }
    
    std::string to_string() const {
        std::stringstream ss;
        bool first = true;
        for (const auto& [name, meter] : meters_) {
            if (!first) ss << delimiter_;
            ss << name << ": " << meter.to_string();
            first = false;
        }
        return ss.str();
    }
    
private:
    struct SmoothedValue {
        double total = 0.0;
        double avg = 0.0;
        double max_val = -std::numeric_limits<double>::max();
        int count = 0;
        int window_size = 20;
        std::deque<double> values;
        
        void update(double value, int n = 1) {
            values.push_back(value);
            if (values.size() > window_size) {
                values.pop_front();
            }
            
            total += value * n;
            count += n;
            avg = total / count;
            max_val = std::max(max_val, value);
        }
        
        double median() const {
            std::vector<double> sorted(values.begin(), values.end());
            std::sort(sorted.begin(), sorted.end());
            if (sorted.empty()) return 0.0;
            return sorted[sorted.size() / 2];
        }
        
        std::string to_string() const {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4) << median() 
               << " (" << avg << ")";
            return ss.str();
        }
    };
    
    std::string delimiter_;
    std::map<std::string, SmoothedValue> meters_;
};

// Utility functions
inline double get_world_size() {
    // For single GPU training
    return 1.0;
}

inline std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}