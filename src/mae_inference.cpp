#include "mae_inference.hpp"
#include <torch/script.h>
#include <iostream>
#include <stdexcept>

MAEInference::MAEInference(const std::string& model_path, torch::Device device) 
    : device_(device) {
    try {
        // Load the TorchScript model
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
        
        std::cout << "Loaded MAE model from " << model_path << std::endl;
        std::cout << "Using device: " << device_ << std::endl;
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

torch::Tensor MAEInference::preprocess_image(const cv::Mat& image) {
    cv::Mat img_float, img_resized, img_rgb;
    
    // Ensure image is RGB
    if (image.channels() == 1) {
        cv::cvtColor(image, img_rgb, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, img_rgb, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);
    }
    
    // Resize to model input size
    cv::resize(img_rgb, img_resized, cv::Size(img_size_, img_size_));
    
    // Convert to float and normalize to [0, 1]
    img_resized.convertTo(img_float, CV_32FC3, 1.0 / 255.0);
    
    // Convert to tensor
    torch::Tensor tensor = torch::from_blob(
        img_float.data, 
        {1, img_size_, img_size_, 3}, 
        torch::kFloat32
    ).clone();
    
    // Change from HWC to CHW format
    tensor = tensor.permute({0, 3, 1, 2});
    
    // Normalize with ImageNet statistics
    for (int c = 0; c < 3; ++c) {
        tensor[0][c] = (tensor[0][c] - mean_[c]) / std_[c];
    }
    
    return tensor.to(device_);
}

torch::Tensor MAEInference::infer(const cv::Mat& image) {
    torch::NoGradGuard no_grad;
    
    // Preprocess the image
    auto input = preprocess_image(image);
    
    // Create inputs vector for the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    inputs.push_back(mask_ratio_);  // Add mask ratio as second argument
    
    // Run inference
    auto output = model_.forward(inputs);
    
    // Extract the loss, pred, and mask from the output tuple
    // For inference, we typically want the prediction
    if (output.isTuple()) {
        auto tuple = output.toTuple();
        if (tuple->elements().size() >= 2) {
            return tuple->elements()[1].toTensor();  // Return prediction
        }
    }
    
    return output.toTensor();
}

torch::Tensor MAEInference::infer_batch(const std::vector<cv::Mat>& images) {
    torch::NoGradGuard no_grad;
    
    if (images.empty()) {
        throw std::invalid_argument("Empty image batch");
    }
    
    // Preprocess all images
    std::vector<torch::Tensor> tensors;
    for (const auto& img : images) {
        tensors.push_back(preprocess_image(img).squeeze(0));  // Remove batch dimension
    }
    
    // Stack into batch
    auto batch = torch::stack(tensors, 0).to(device_);
    
    // Create inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch);
    inputs.push_back(mask_ratio_);
    
    // Run inference
    auto output = model_.forward(inputs);
    
    // Extract prediction from tuple if necessary
    if (output.isTuple()) {
        auto tuple = output.toTuple();
        if (tuple->elements().size() >= 2) {
            return tuple->elements()[1].toTensor();
        }
    }
    
    return output.toTensor();
}

cv::Mat MAEInference::get_reconstruction(const torch::Tensor& output) {
    // Ensure tensor is on CPU and contiguous
    auto tensor = output.to(torch::kCPU).contiguous();
    
    // Get the first image if batch
    if (tensor.dim() == 4) {
        tensor = tensor[0];
    }
    
    // Denormalize
    auto denorm = tensor.clone();
    for (int c = 0; c < 3; ++c) {
        denorm[c] = denorm[c] * std_[c] + mean_[c];
    }
    
    // Clamp to [0, 1]
    denorm = torch::clamp(denorm, 0.0, 1.0);
    
    // Convert to uint8
    denorm = (denorm * 255).to(torch::kUInt8);
    
    // Change from CHW to HWC
    denorm = denorm.permute({1, 2, 0});
    
    // Create cv::Mat
    cv::Mat img(img_size_, img_size_, CV_8UC3, denorm.data_ptr());
    cv::Mat img_bgr;
    cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
    
    return img_bgr.clone();
}

cv::Mat MAEInference::reconstruct_image(const cv::Mat& image, float mask_ratio) {
    // Set mask ratio for this reconstruction
    float old_ratio = mask_ratio_;
    mask_ratio_ = mask_ratio;
    
    // Run inference
    auto output = infer(image);
    
    // Restore original mask ratio
    mask_ratio_ = old_ratio;
    
    // Get reconstruction
    return get_reconstruction(output);
}