#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <vector>

class MAEInference {
public:
    // Constructor that loads the TorchScript model
    explicit MAEInference(const std::string& model_path, 
                         torch::Device device = torch::kCPU);
    
    // Preprocess image for inference
    torch::Tensor preprocess_image(const cv::Mat& image);
    
    // Run inference on a single image
    torch::Tensor infer(const cv::Mat& image);
    
    // Run inference on batch of images
    torch::Tensor infer_batch(const std::vector<cv::Mat>& images);
    
    // Get reconstructed image as cv::Mat
    cv::Mat get_reconstruction(const torch::Tensor& output);
    
    // Process image and return reconstruction
    cv::Mat reconstruct_image(const cv::Mat& image, float mask_ratio = 0.75);
    
    // Get model device
    torch::Device get_device() const { return device_; }
    
    // Set mask ratio for inference
    void set_mask_ratio(float ratio) { mask_ratio_ = ratio; }
    
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    int img_size_ = 224;
    float mask_ratio_ = 0.75;
    
    // Image normalization parameters (ImageNet)
    std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
};