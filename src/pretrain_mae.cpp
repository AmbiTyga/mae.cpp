#include "mae_model.hpp"
#include "data_loader.hpp"
#include "pretrain_utils.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <cmath>

using json = nlohmann::json;

// Enhanced logger class for dual output with console clearing
class PretrainLogger {
public:
    PretrainLogger(const std::string& filename, int64_t console_clear_freq = 100) 
        : console_line_count_(0), log_filename_(filename), console_clear_freq_(console_clear_freq) {
        log_file_.open(filename, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Warning: Could not open log file " << filename << std::endl;
        }
    }
    
    ~PretrainLogger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
    
    template<typename T>
    PretrainLogger& operator<<(const T& value) {
        std::cout << value;
        if (log_file_.is_open()) {
            log_file_ << value;
            // Don't flush on every write - only on newline
        }
        return *this;
    }
    
    PretrainLogger& operator<<(std::ostream& (*pf)(std::ostream&)) {
        std::cout << pf;
        if (log_file_.is_open()) {
            log_file_ << pf;
            log_file_.flush();
        }
        console_line_count_++;
        
        if (console_line_count_ >= console_clear_freq_) {
            clearConsole();
            console_line_count_ = 0;
        }
        return *this;
    }
    
    void clearConsole() {
        #ifdef _WIN32
            system("cls");
        #else
            system("clear");
        #endif
        std::cout << "Pretraining in progress... Check " << log_filename_ << " for full logs" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    const std::string& getFilename() const { return log_filename_; }
    
    void copyLogToCheckpoint(const std::string& checkpoint_dir) {
        if (log_file_.is_open()) {
            log_file_.flush();
            std::string dest_path = checkpoint_dir + "/" + log_filename_;
            std::filesystem::copy_file(log_filename_, dest_path, 
                                      std::filesystem::copy_options::overwrite_existing);
        }
    }
    
private:
    std::ofstream log_file_;
    std::string log_filename_;
    size_t console_line_count_;
    int64_t console_clear_freq_;
};

// Save checkpoint with logs and metadata
void save_checkpoint_with_logs(const MaskedAutoencoderViT& model, 
                              const torch::optim::Optimizer& optimizer,
                              int64_t epoch,
                              int64_t step,
                              double loss,
                              double lr,
                              const std::string& base_dir,
                              const std::string& suffix,
                              PretrainLogger& logger,
                              const PretrainConfig& config) {
    // Create directory for this checkpoint
    std::string checkpoint_dir = base_dir + "/" + suffix;
    std::filesystem::create_directories(checkpoint_dir);
    
    // Save model checkpoint
    std::string model_path = checkpoint_dir + "/model.pt";
    torch::serialize::OutputArchive archive;
    model->save(archive);
    
    // Save training state
    archive.write("epoch", torch::tensor(epoch));
    archive.write("step", torch::tensor(step));
    archive.write("loss", torch::tensor(loss));
    archive.write("lr", torch::tensor(lr));
    
    archive.save_to(model_path);
    
    // Save training metadata as JSON
    json metadata;
    metadata["epoch"] = epoch;
    metadata["step"] = step;
    metadata["loss"] = loss;
    metadata["lr"] = lr;
    metadata["timestamp"] = get_timestamp();
    
    std::ofstream metadata_file(checkpoint_dir + "/metadata.json");
    metadata_file << metadata.dump(4);
    metadata_file.close();
    
    // Save config
    std::ofstream config_file(checkpoint_dir + "/config.json");
    config_file << std::ifstream(base_dir + "/config.json").rdbuf();
    config_file.close();
    
    // Copy current log to checkpoint directory
    logger.copyLogToCheckpoint(checkpoint_dir);
    
    logger << "Saved checkpoint to " << checkpoint_dir << " (epoch: " << epoch 
           << ", step: " << step << ", loss: " << std::fixed << std::setprecision(4) 
           << loss << ", lr: " << std::scientific << std::setprecision(2) << lr << ")" << std::endl;
}

// Load checkpoint from file
bool load_checkpoint(MaskedAutoencoderViT& model,
                    torch::optim::Optimizer& optimizer,
                    int64_t& epoch,
                    int64_t& global_step,
                    const std::string& filepath,
                    PretrainLogger& logger) {
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    torch::serialize::InputArchive archive;
    archive.load_from(filepath);
    
    model->load(archive);
    
    // Load training state
    torch::Tensor epoch_tensor, step_tensor;
    archive.read("epoch", epoch_tensor);
    archive.read("step", step_tensor);
    epoch = epoch_tensor.item().toLong();
    global_step = step_tensor.item().toLong();
    
    logger << "Loaded checkpoint from " << filepath 
           << " (epoch: " << epoch << ", step: " << global_step << ")" << std::endl;
    return true;
}

// Load checkpoint from directory by reading metadata
bool load_checkpoint_from_dir(MaskedAutoencoderViT& model,
                             torch::optim::Optimizer& optimizer,
                             int64_t& epoch,
                             int64_t& global_step,
                             const std::string& dir_path,
                             PretrainLogger& logger) {
    // Check if directory exists
    if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
        logger << "Error: Directory '" << dir_path << "' not found or is not a directory." << std::endl;
        return false;
    }
    
    // Check for metadata.json
    std::string metadata_path = dir_path + "/metadata.json";
    if (std::filesystem::exists(metadata_path)) {
        // Read metadata to get epoch and step
        std::ifstream metadata_file(metadata_path);
        json metadata;
        metadata_file >> metadata;
        metadata_file.close();
        
        epoch = metadata["epoch"];
        global_step = metadata["step"];
        
        logger << "Found checkpoint metadata - Epoch: " << epoch 
               << ", Step: " << global_step << std::endl;
    }
    
    // Load model from model.pt
    std::string model_path = dir_path + "/model.pt";
    if (!std::filesystem::exists(model_path)) {
        logger << "Error: Model file '" << model_path << "' not found in checkpoint directory." << std::endl;
        return false;
    }
    
    return load_checkpoint(model, optimizer, epoch, global_step, model_path, logger);
}

// Enhanced training function with proper augmentation
template<typename DataLoader>
void train_one_epoch(MaskedAutoencoderViT& model,
                    DataLoader& data_loader,
                    torch::optim::Optimizer& optimizer,
                    const PretrainConfig& config,
                    int64_t epoch,
                    int64_t& global_step,
                    size_t iterations_per_epoch,
                    CosineAnnealingWarmupScheduler& lr_scheduler,
                    MetricLogger& metric_logger,
                    PretrainLogger& logger) {
    model->train();
    
    metric_logger.update("epoch", epoch);
    
    auto epoch_start = std::chrono::high_resolution_clock::now();
    auto data_start = std::chrono::high_resolution_clock::now();
    
    size_t batch_idx = 0;
    
    for (auto& batch : *data_loader) {
        auto data_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - data_start).count();
        
        // Update learning rate per iteration (following MAE paper)
        double lr = lr_scheduler.get_lr(epoch + static_cast<double>(batch_idx) / iterations_per_epoch);
        for (auto& param_group : optimizer.param_groups()) {
            param_group.options().set_lr(lr);
        }
        
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        // Move data to device
        auto images = batch.data.to(config.device);
        
        // Zero gradients from previous iteration
        optimizer.zero_grad();
        
        // Forward pass - compute loss with autograd enabled
        auto [loss, pred, mask] = model->forward(images, config.mask_ratio);
        
        // Backward pass - compute gradients
        loss.backward();
        
        // Gradient clipping for stability
        torch::nn::utils::clip_grad_norm_(model->parameters(), config.gradient_clip);
        
        // Update parameters using computed gradients
        optimizer.step();
        
        // Clear intermediate tensors to save memory (gradients are already applied)
        pred = torch::Tensor();
        mask = torch::Tensor();
        
        // Synchronize CUDA if using GPU
        if (config.device.type() == torch::kCUDA) {
            torch::cuda::synchronize();
        }
        
        auto iter_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - iter_start).count();
        
        // Update metrics
        double loss_value = loss.item().toDouble();
        
        // Verify loss is valid
        if (std::isnan(loss_value) || std::isinf(loss_value)) {
            logger << "WARNING: Invalid loss detected: " << loss_value 
                   << " at step " << global_step << std::endl;
            logger << "Skipping this iteration..." << std::endl;
            continue;
        }
        
        metric_logger.update("loss", loss_value);
        metric_logger.update("lr", lr);
        metric_logger.update("data_time", data_time);
        metric_logger.update("iter_time", iter_time);
        
        global_step++;
        batch_idx++;
        
        // Save checkpoint at specified steps
        if (config.save_freq_steps > 0 && global_step % config.save_freq_steps == 0) {
            std::string suffix = "step-" + std::to_string(global_step);
            save_checkpoint_with_logs(model, optimizer, epoch, global_step, loss_value, lr,
                                    config.checkpoint_dir, suffix, logger, config);
        }
        
        // Print progress
        if (batch_idx % config.print_freq == 0) {
            double samples_per_sec = config.batch_size / iter_time;
            double eta_seconds = (iterations_per_epoch - batch_idx) * iter_time;
            
            logger << "Epoch: [" << epoch << "][" << batch_idx << "/" << iterations_per_epoch << "]  "
                   << "eta: " << std::fixed << std::setprecision(0) << eta_seconds << "s  "
                   << "loss: " << metric_logger.get_str("loss") << "  "
                   << "lr: " << std::scientific << std::setprecision(2) << lr << "  "
                   << "time: " << std::fixed << std::setprecision(3) << iter_time << "  "
                   << "data: " << std::fixed << std::setprecision(3) << data_time << "  "
                   << "speed: " << std::fixed << std::setprecision(0) << samples_per_sec << " samples/s"
                   << std::endl;
            
            // Periodically check gradient norms for debugging
            if (batch_idx % (config.print_freq * 5) == 0 && batch_idx > 0) {
                double total_grad_norm = 0.0;
                for (const auto& p : model->parameters()) {
                    if (p.grad().defined()) {
                        auto grad_norm = p.grad().norm(2).item().toDouble();
                        total_grad_norm += grad_norm * grad_norm;
                    }
                }
                total_grad_norm = std::sqrt(total_grad_norm);
                logger << "  Gradient norm: " << std::fixed << std::setprecision(4) 
                       << total_grad_norm << std::endl;
            }
            
            // Periodically clear CUDA cache to prevent memory fragmentation
            // Note: empty_cache is not available in all LibTorch versions
            // Synchronization alone helps with memory management
        }
        
        data_start = std::chrono::high_resolution_clock::now();
    }
    
    // Epoch statistics
    auto epoch_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - epoch_start).count();
    
    logger << "Epoch [" << epoch << "] completed in " << std::fixed << std::setprecision(1) 
           << epoch_time/60.0 << " min  "
           << "Average loss: " << metric_logger.get_avg("loss") << std::endl;
    logger << "========================================" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file.json>" << std::endl;
        std::cerr << "Example: " << argv[0] << " configs/mae_pretrain_vit_base.json" << std::endl;
        return 1;
    }
    
    // Set environment variables for better performance
    setenv("OMP_NUM_THREADS", "1", 1);  // Prevent OpenMP from spawning too many threads
    setenv("MKL_NUM_THREADS", "1", 1);  // Control MKL threads
    
    // Set PyTorch inter-op threads
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    
    // Load configuration
    PretrainConfig config;
    try {
        config = PretrainConfig::from_json(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return 1;
    }
    
    // Set random seed
    torch::manual_seed(config.seed);
    
    // Create logger
    std::string log_filename = "mae_pretrain_" + get_timestamp() + ".log";
    PretrainLogger logger(log_filename, config.console_clear_freq);
    
    // Copy config file to checkpoint directory
    std::filesystem::create_directories(config.checkpoint_dir);
    std::filesystem::copy_file(argv[1], config.checkpoint_dir + "/config.json",
                              std::filesystem::copy_options::overwrite_existing);
    
    logger << "MAE Pretraining Configuration" << std::endl;
    logger << "=============================" << std::endl;
    logger << "Model: " << config.model_type << std::endl;
    logger << "Device: " << config.device << std::endl;
    logger << "Batch size: " << config.batch_size << std::endl;
    logger << "Base LR: " << config.base_lr << std::endl;
    logger << "Min LR: " << config.min_lr << std::endl;
    logger << "Weight decay: " << config.weight_decay << std::endl;
    logger << "Epochs: " << config.epochs << std::endl;
    logger << "Warmup epochs: " << config.warmup_epochs << std::endl;
    logger << "Data path: " << config.data_path << std::endl;
    logger << "Effective batch size: " << config.batch_size * get_world_size() << std::endl;
    logger << "=============================" << std::endl << std::endl;
    
    // Create model
    MaskedAutoencoderViT model;
    if (config.model_type == "mae_vit_base_patch16") {
        model = mae_vit_base_patch16_dec512d8b(config.norm_pix_loss);
    } else if (config.model_type == "mae_vit_large_patch16") {
        model = mae_vit_large_patch16_dec512d8b(config.norm_pix_loss);
    } else if (config.model_type == "mae_vit_huge_patch14") {
        model = mae_vit_huge_patch14_dec512d8b(config.norm_pix_loss);
    } else {
        std::cerr << "Unknown model type: " << config.model_type << std::endl;
        return 1;
    }
    
    model->to(config.device);
    
    // Count parameters
    int64_t param_count = 0;
    for (const auto& p : model->parameters()) {
        param_count += p.numel();
    }
    logger << "Model parameters: " << param_count / 1e6 << "M" << std::endl << std::endl;
    
    // Create dataset with proper augmentation
    ImageFolderDataset dataset_raw(config.data_path, config.input_size);
    size_t dataset_size = dataset_raw.size().value();
    size_t iterations_per_epoch = dataset_size / config.batch_size;
    
    auto dataset = dataset_raw.map(torch::data::transforms::Stack<>());
    
    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(config.batch_size)
            .workers(config.num_workers)
            .drop_last(true)
    );
    
    logger << "Dataset size: " << dataset_size << " images" << std::endl;
    logger << "Iterations per epoch: " << iterations_per_epoch << std::endl << std::endl;
    
    // Create optimizer
    auto optimizer = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(config.base_lr)
            .weight_decay(config.weight_decay)
            .betas({config.adam_beta1, config.adam_beta2})
    );
    
    // Create learning rate scheduler
    CosineAnnealingWarmupScheduler lr_scheduler(optimizer, config.warmup_epochs, 
                                               config.epochs, config.base_lr, config.min_lr);
    
    // Load checkpoint if exists
    int64_t start_epoch = config.start_epoch;
    int64_t global_step = 0;
    
    if (!config.resume.empty()) {
        std::string checkpoint_path = config.resume;
        
        // If resume path doesn't contain directory separator, assume it's in checkpoint_dir
        if (checkpoint_path.find('/') == std::string::npos && checkpoint_path.find('\\') == std::string::npos) {
            checkpoint_path = config.checkpoint_dir + "/" + checkpoint_path;
        }
        
        // Check if it's a directory or a file
        if (std::filesystem::exists(checkpoint_path)) {
            if (std::filesystem::is_directory(checkpoint_path)) {
                // Load from directory using metadata
                if (!load_checkpoint_from_dir(model, optimizer, start_epoch, global_step, checkpoint_path, logger)) {
                    logger << "Failed to load checkpoint from directory. Starting from scratch..." << std::endl;
                    start_epoch = config.start_epoch;
                    global_step = 0;
                }
            } else {
                // Load from file directly
                if (!load_checkpoint(model, optimizer, start_epoch, global_step, checkpoint_path, logger)) {
                    logger << "Failed to load checkpoint file. Starting from scratch..." << std::endl;
                    start_epoch = config.start_epoch;
                    global_step = 0;
                }
            }
        } else {
            logger << "Warning: Resume checkpoint '" << checkpoint_path << "' not found." << std::endl;
            logger << "Starting pretraining from scratch..." << std::endl;
            logger << "========================================" << std::endl;
        }
    } else if (config.auto_resume) {
        // First try to find the latest checkpoint directory
        std::string latest_dir;
        std::filesystem::file_time_type latest_time;
        bool found_checkpoint = false;
        
        // Look for step-* and epoch-* directories
        if (std::filesystem::exists(config.checkpoint_dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(config.checkpoint_dir)) {
                if (entry.is_directory()) {
                    std::string dirname = entry.path().filename().string();
                    if (dirname.find("step-") == 0 || dirname.find("epoch-") == 0) {
                        auto mod_time = std::filesystem::last_write_time(entry);
                        if (!found_checkpoint || mod_time > latest_time) {
                            latest_time = mod_time;
                            latest_dir = entry.path().string();
                            found_checkpoint = true;
                        }
                    }
                }
            }
        }
        
        if (found_checkpoint) {
            logger << "Auto-resuming from latest checkpoint: " << latest_dir << std::endl;
            if (!load_checkpoint_from_dir(model, optimizer, start_epoch, global_step, latest_dir, logger)) {
                logger << "Failed to load checkpoint. Starting from scratch..." << std::endl;
                start_epoch = config.start_epoch;
                global_step = 0;
            }
        } else {
            // Fallback to latest.pt if it exists
            std::string latest_checkpoint = config.checkpoint_dir + "/latest.pt";
            if (std::filesystem::exists(latest_checkpoint)) {
                load_checkpoint(model, optimizer, start_epoch, global_step, latest_checkpoint, logger);
            } else {
                logger << "No checkpoint found for auto-resume." << std::endl;
                logger << "Starting pretraining from scratch..." << std::endl;
                logger << "========================================" << std::endl;
            }
        }
    }
    
    // Create metric logger
    MetricLogger metric_logger("  ");
    
    // Training loop
    for (int64_t epoch = start_epoch; epoch < config.epochs; ++epoch) {
        // Train for one epoch
        train_one_epoch(model, data_loader, optimizer, config, epoch, global_step,
                       iterations_per_epoch, lr_scheduler, metric_logger, logger);
        
        // Save checkpoint at epoch intervals
        if ((epoch + 1) % config.save_freq_epochs == 0 || epoch == config.epochs - 1) {
            // Save epoch checkpoint with logs
            std::string suffix = "epoch-" + std::to_string(epoch);
            double current_lr = lr_scheduler.get_lr(epoch);
            save_checkpoint_with_logs(model, optimizer, epoch, global_step, 
                                    metric_logger.get_avg("loss"), current_lr,
                                    config.checkpoint_dir, suffix, logger, config);
            
            // Also save as latest
            std::string latest_path = config.checkpoint_dir + "/latest.pt";
            torch::serialize::OutputArchive archive;
            model->save(archive);
            archive.write("epoch", torch::tensor(epoch));
            archive.write("step", torch::tensor(global_step));
            archive.save_to(latest_path);
        }
    }
    
    logger << "\nPretraining completed!" << std::endl;
    
    // Save final model
    std::string final_checkpoint = config.checkpoint_dir + "/final.pt";
    torch::serialize::OutputArchive final_archive;
    model->save(final_archive);
    final_archive.save_to(final_checkpoint);
    logger << "Saved final model to " << final_checkpoint << std::endl;
    
    return 0;
}