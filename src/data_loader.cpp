#include "data_loader.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

ImageFolderDataset::ImageFolderDataset(const std::string& root_path, 
                                     int64_t img_size,
                                     const std::vector<double>& mean,
                                     const std::vector<double>& std) 
    : img_size_(img_size), mean_(mean), std_(std) {
    load_image_paths(root_path);
}

void ImageFolderDataset::load_image_paths(const std::string& root_path) {
    std::vector<std::string> class_names;
    
    // Get all subdirectories (classes)
    for (const auto& entry : fs::directory_iterator(root_path)) {
        if (entry.is_directory()) {
            class_names.push_back(entry.path().filename().string());
        }
    }
    
    // Sort for consistent ordering
    std::sort(class_names.begin(), class_names.end());
    
    // Load images from each class directory
    for (size_t class_idx = 0; class_idx < class_names.size(); ++class_idx) {
        std::string class_path = root_path + "/" + class_names[class_idx];
        
        for (const auto& entry : fs::directory_iterator(class_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    image_paths_.push_back(entry.path().string());
                    labels_.push_back(class_idx);
                }
            }
        }
    }
    
    std::cout << "Loaded " << image_paths_.size() << " images from " 
              << class_names.size() << " classes" << std::endl;
}

torch::Tensor ImageFolderDataset::transform(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(img_size_, img_size_));
    
    // Convert BGR to RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    // Convert to tensor and normalize
    torch::Tensor tensor = torch::from_blob(resized.data, 
                                           {resized.rows, resized.cols, 3}, 
                                           torch::kByte);
    tensor = tensor.permute({2, 0, 1}).to(torch::kFloat32) / 255.0;
    
    // Normalize with ImageNet stats
    for (int c = 0; c < 3; ++c) {
        tensor[c] = (tensor[c] - mean_[c]) / std_[c];
    }
    
    return tensor;
}

torch::data::Example<> ImageFolderDataset::get(size_t index) {
    cv::Mat img = cv::imread(image_paths_[index]);
    
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_paths_[index]);
    }
    
    torch::Tensor data = transform(img);
    torch::Tensor label = torch::tensor(labels_[index], torch::kLong);
    
    return {data, label};
}

torch::optional<size_t> ImageFolderDataset::size() const {
    return image_paths_.size();
}

// RandomResizedCrop implementation
RandomResizedCrop::RandomResizedCrop(int64_t size, 
                                   const std::pair<double, double>& scale,
                                   const std::pair<double, double>& ratio)
    : size_(size), scale_(scale), ratio_(ratio), 
      gen_(std::random_device{}()),
      scale_dist_(scale.first, scale.second),
      ratio_dist_(std::log(ratio.first), std::log(ratio.second)) {}

cv::Mat RandomResizedCrop::operator()(const cv::Mat& img) {
    int height = img.rows;
    int width = img.cols;
    double area = height * width;
    
    for (int attempt = 0; attempt < 10; ++attempt) {
        double target_area = area * scale_dist_(gen_);
        double aspect_ratio = std::exp(ratio_dist_(gen_));
        
        int w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
        int h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));
        
        if (w <= width && h <= height) {
            std::uniform_int_distribution<> x_dist(0, width - w);
            std::uniform_int_distribution<> y_dist(0, height - h);
            
            int x = x_dist(gen_);
            int y = y_dist(gen_);
            
            cv::Mat cropped = img(cv::Rect(x, y, w, h));
            cv::Mat resized;
            cv::resize(cropped, resized, cv::Size(size_, size_));
            
            return resized;
        }
    }
    
    // Fallback to central crop
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(size_, size_));
    return resized;
}

// RandomHorizontalFlip implementation
RandomHorizontalFlip::RandomHorizontalFlip(double p) 
    : p_(p), gen_(std::random_device{}()), dist_(0.0, 1.0) {}

cv::Mat RandomHorizontalFlip::operator()(const cv::Mat& img) {
    if (dist_(gen_) < p_) {
        cv::Mat flipped;
        cv::flip(img, flipped, 1);
        return flipped;
    }
    return img;
}