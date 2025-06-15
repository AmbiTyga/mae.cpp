#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <random>

// Custom dataset for image loading
class ImageFolderDataset : public torch::data::Dataset<ImageFolderDataset> {
public:
    ImageFolderDataset(const std::string& root_path, 
                      int64_t img_size = 224,
                      const std::vector<double>& mean = {0.485, 0.456, 0.406},
                      const std::vector<double>& std = {0.229, 0.224, 0.225});
    
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
    
private:
    std::vector<std::string> image_paths_;
    std::vector<int64_t> labels_;
    int64_t img_size_;
    std::vector<double> mean_;
    std::vector<double> std_;
    
    void load_image_paths(const std::string& root_path);
    torch::Tensor transform(const cv::Mat& img);
};

// Data augmentation utilities
class RandomResizedCrop {
public:
    RandomResizedCrop(int64_t size, 
                     const std::pair<double, double>& scale = {0.2, 1.0},
                     const std::pair<double, double>& ratio = {0.75, 1.33});
    
    cv::Mat operator()(const cv::Mat& img);
    
private:
    int64_t size_;
    std::pair<double, double> scale_;
    std::pair<double, double> ratio_;
    std::mt19937 gen_;
    std::uniform_real_distribution<> scale_dist_;
    std::uniform_real_distribution<> ratio_dist_;
};

class RandomHorizontalFlip {
public:
    RandomHorizontalFlip(double p = 0.5);
    cv::Mat operator()(const cv::Mat& img);
    
private:
    double p_;
    std::mt19937 gen_;
    std::uniform_real_distribution<> dist_;
};