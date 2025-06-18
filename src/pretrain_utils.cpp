#include "pretrain_utils.hpp"
#include <iostream>
#include <stdexcept>

PretrainConfig PretrainConfig::from_json(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_path);
    }
    
    json j;
    config_file >> j;
    config_file.close();
    
    PretrainConfig config;
    
    // Model config
    config.model_type = j["model"]["type"];
    config.norm_pix_loss = j["model"]["norm_pix_loss"];
    config.mask_ratio = j["model"]["mask_ratio"];
    
    // Optimization config
    config.batch_size = j["optimization"]["batch_size"];
    config.base_lr = j["optimization"]["base_lr"];
    config.min_lr = j["optimization"]["min_lr"];
    config.weight_decay = j["optimization"]["weight_decay"];
    config.optimizer_type = j["optimization"]["optimizer"];
    config.adam_beta1 = j["optimization"]["adam_beta1"];
    config.adam_beta2 = j["optimization"]["adam_beta2"];
    config.gradient_clip = j["optimization"]["gradient_clip"];
    
    // Schedule config
    config.epochs = j["schedule"]["epochs"];
    config.warmup_epochs = j["schedule"]["warmup_epochs"];
    config.start_epoch = j["schedule"]["start_epoch"];
    
    // Data config
    config.data_path = j["data"]["data_path"];
    config.input_size = j["data"]["input_size"];
    config.num_workers = j["data"]["num_workers"];
    
    // Augmentation config
    auto scale = j["data"]["augmentation"]["random_resized_crop"]["scale"];
    config.crop_scale = {scale[0], scale[1]};
    config.interpolation = j["data"]["augmentation"]["random_resized_crop"]["interpolation"];
    config.random_flip = j["data"]["augmentation"]["random_horizontal_flip"];
    
    auto mean = j["data"]["augmentation"]["normalize"]["mean"];
    config.normalize_mean = {mean[0], mean[1], mean[2]};
    
    auto std = j["data"]["augmentation"]["normalize"]["std"];
    config.normalize_std = {std[0], std[1], std[2]};
    
    // Checkpointing config
    config.checkpoint_dir = j["checkpointing"]["checkpoint_dir"];
    config.save_freq_epochs = j["checkpointing"]["save_freq_epochs"];
    config.save_freq_steps = j["checkpointing"]["save_freq_steps"];
    config.keep_last_n = j["checkpointing"]["keep_last_n"];
    
    // Logging config
    config.print_freq = j["logging"]["print_freq"];
    config.console_clear_freq = j["logging"]["console_clear_freq"];
    
    // Misc config
    config.seed = j["misc"]["seed"];
    config.resume = j["misc"]["resume"];
    config.auto_resume = j["misc"]["auto_resume"];
    
    return config;
}