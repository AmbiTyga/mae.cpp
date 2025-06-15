#include "mae_model.hpp"
#include "data_loader.hpp"
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <iomanip>

// Training configuration
struct TrainingConfig {
    // Model parameters
    std::string model_type = "mae_vit_base_patch16";
    bool norm_pix_loss = true;
    
    // Training parameters
    int64_t batch_size = 64;
    int64_t epochs = 400;
    double learning_rate = 1.5e-4;
    double weight_decay = 0.05;
    double mask_ratio = 0.75;
    
    // Warmup parameters
    int64_t warmup_epochs = 40;
    double min_lr = 0.0;
    
    // Data parameters
    std::string data_path = "./data/imagenet/train";
    int64_t img_size = 224;
    int64_t num_workers = 4;
    
    // Checkpointing
    std::string checkpoint_dir = "./checkpoints";
    int64_t save_freq = 20;
    
    // Device
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
};

// Learning rate scheduler with warmup
class CosineAnnealingWarmupLR {
public:
    CosineAnnealingWarmupLR(torch::optim::Optimizer& optimizer, 
                           int64_t warmup_epochs, 
                           int64_t total_epochs,
                           double base_lr,
                           double min_lr = 0.0)
        : optimizer_(optimizer), 
          warmup_epochs_(warmup_epochs), 
          total_epochs_(total_epochs),
          base_lr_(base_lr),
          min_lr_(min_lr) {}
    
    void step(int64_t epoch) {
        double lr;
        if (epoch < warmup_epochs_) {
            // Linear warmup
            lr = base_lr_ * epoch / warmup_epochs_;
        } else {
            // Cosine annealing
            double progress = static_cast<double>(epoch - warmup_epochs_) / 
                            (total_epochs_ - warmup_epochs_);
            lr = min_lr_ + (base_lr_ - min_lr_) * 0.5 * 
                 (1 + std::cos(M_PI * progress));
        }
        
        for (auto& param_group : optimizer_.param_groups()) {
            param_group.options().set_lr(lr);
        }
    }
    
    double get_lr() const {
        return optimizer_.param_groups()[0].options().get_lr();
    }
    
private:
    torch::optim::Optimizer& optimizer_;
    int64_t warmup_epochs_;
    int64_t total_epochs_;
    double base_lr_;
    double min_lr_;
};

// Utility function to save checkpoint
void save_checkpoint(const MaskedAutoencoderViT& model, 
                    const torch::optim::Optimizer& optimizer,
                    int64_t epoch,
                    double loss,
                    const std::string& filepath) {
    torch::serialize::OutputArchive archive;
    model->save(archive);
    
    // Save optimizer state
    auto opt_state = optimizer.state_dict();
    archive.write("optimizer", opt_state);
    
    // Save epoch and loss
    archive.write("epoch", torch::tensor(epoch));
    archive.write("loss", torch::tensor(loss));
    
    archive.save_to(filepath);
    std::cout << "Saved checkpoint to " << filepath << std::endl;
}

// Utility function to load checkpoint
bool load_checkpoint(MaskedAutoencoderViT& model,
                    torch::optim::Optimizer& optimizer,
                    int64_t& epoch,
                    const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    torch::serialize::InputArchive archive;
    archive.load_from(filepath);
    
    model->load(archive);
    
    // Load optimizer state
    torch::OrderedDict<std::string, torch::Tensor> opt_state;
    archive.read("optimizer", opt_state);
    optimizer.load_state_dict(opt_state);
    
    // Load epoch
    torch::Tensor epoch_tensor;
    archive.read("epoch", epoch_tensor);
    epoch = epoch_tensor.item<int64_t>();
    
    std::cout << "Loaded checkpoint from " << filepath << " (epoch " << epoch << ")" << std::endl;
    return true;
}

// Training function
void train_epoch(MaskedAutoencoderViT& model,
                torch::data::DataLoaderBase<torch::data::datasets::MapDataset<ImageFolderDataset, 
                                                                              torch::data::transforms::Stack<>>>& data_loader,
                torch::optim::Optimizer& optimizer,
                const TrainingConfig& config,
                int64_t epoch) {
    model->train();
    
    double total_loss = 0.0;
    size_t batch_count = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (auto& batch : data_loader) {
        auto data = batch.data.to(config.device);
        auto target = batch.target.to(config.device);
        
        optimizer.zero_grad();
        
        // Forward pass
        auto [loss, pred, mask] = model->forward(data, config.mask_ratio);
        
        // Backward pass
        loss.backward();
        
        // Gradient clipping
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        
        optimizer.step();
        
        total_loss += loss.item<double>();
        batch_count++;
        
        // Print progress
        if (batch_count % 100 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            std::cout << "Epoch [" << epoch << "] Batch [" << batch_count << "] "
                     << "Loss: " << std::fixed << std::setprecision(4) << loss.item<double>() 
                     << " Time: " << duration.count() << "s" << std::endl;
        }
    }
    
    double avg_loss = total_loss / batch_count;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    
    std::cout << "Epoch [" << epoch << "] completed - "
              << "Average Loss: " << std::fixed << std::setprecision(4) << avg_loss
              << " Time: " << duration.count() << " minutes" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    TrainingConfig config;
    
    if (argc > 1) {
        config.data_path = argv[1];
    }
    if (argc > 2) {
        config.batch_size = std::stoi(argv[2]);
    }
    if (argc > 3) {
        config.epochs = std::stoi(argv[3]);
    }
    
    std::cout << "MAE Training Configuration:" << std::endl;
    std::cout << "  Model: " << config.model_type << std::endl;
    std::cout << "  Device: " << config.device << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Epochs: " << config.epochs << std::endl;
    std::cout << "  Learning rate: " << config.learning_rate << std::endl;
    std::cout << "  Mask ratio: " << config.mask_ratio << std::endl;
    std::cout << "  Data path: " << config.data_path << std::endl;
    std::cout << std::endl;
    
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
    std::cout << "Model parameters: " << param_count / 1e6 << "M" << std::endl;
    
    // Create dataset and dataloader
    auto dataset = ImageFolderDataset(config.data_path, config.img_size)
        .map(torch::data::transforms::Stack<>());
    
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(config.batch_size)
            .workers(config.num_workers)
            .drop_last(true)
    );
    
    // Create optimizer
    auto optimizer = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(config.learning_rate)
            .weight_decay(config.weight_decay)
            .betas({0.9, 0.95})
    );
    
    // Create learning rate scheduler
    CosineAnnealingWarmupLR scheduler(optimizer, config.warmup_epochs, 
                                    config.epochs, config.learning_rate, config.min_lr);
    
    // Create checkpoint directory
    std::filesystem::create_directories(config.checkpoint_dir);
    
    // Try to load checkpoint
    int64_t start_epoch = 0;
    std::string checkpoint_path = config.checkpoint_dir + "/mae_latest.pt";
    load_checkpoint(model, optimizer, start_epoch, checkpoint_path);
    
    // Training loop
    for (int64_t epoch = start_epoch; epoch < config.epochs; ++epoch) {
        // Update learning rate
        scheduler.step(epoch);
        std::cout << "\nEpoch " << epoch << " - Learning rate: " 
                  << scheduler.get_lr() << std::endl;
        
        // Train for one epoch
        train_epoch(model, *data_loader, optimizer, config, epoch);
        
        // Save checkpoint
        if ((epoch + 1) % config.save_freq == 0 || epoch == config.epochs - 1) {
            std::string epoch_checkpoint = config.checkpoint_dir + "/mae_epoch_" + 
                                         std::to_string(epoch) + ".pt";
            save_checkpoint(model, optimizer, epoch, 0.0, epoch_checkpoint);
            
            // Also save as latest
            save_checkpoint(model, optimizer, epoch, 0.0, checkpoint_path);
        }
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    
    return 0;
}