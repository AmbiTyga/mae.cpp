#include "mae_model.hpp"
#include <httplib.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cxxopts.hpp>
#include <iomanip>

using json = nlohmann::json;

class MAEServer {
public:
    MAEServer(const std::string& checkpoint_path, const std::string& model_type, 
                    const std::string& host, int port)
        : host_(host), port_(port), device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        
        // Create model based on type
        if (model_type == "mae_vit_base_patch16") {
            model_ = mae_vit_base_patch16_dec512d8b(true);
        } else if (model_type == "mae_vit_large_patch16") {
            model_ = mae_vit_large_patch16_dec512d8b(true);
        } else if (model_type == "mae_vit_huge_patch14") {
            model_ = mae_vit_huge_patch14_dec512d8b(true);
        } else {
            throw std::runtime_error("Unknown model type: " + model_type);
        }
        
        model_->to(device_);
        model_->eval();
        
        // Load checkpoint
        torch::serialize::InputArchive archive;
        archive.load_from(checkpoint_path);
        model_->load(archive);
        
        std::cout << "Loaded MAE model from: " << checkpoint_path << std::endl;
        std::cout << "Model type: " << model_type << std::endl;
        std::cout << "Device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
        
        // Open log file
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream log_filename;
        log_filename << "mae_server_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
        log_file_.open(log_filename.str(), std::ios::app);
        
        if (log_file_.is_open()) {
            log_file_ << "MAE Server started at " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << std::endl;
            log_file_ << "Checkpoint: " << checkpoint_path << std::endl;
            log_file_ << "Model: " << model_type << std::endl;
            log_file_ << "Device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
            log_file_ << "Host: " << host << ":" << port << std::endl;
            log_file_ << "----------------------------------------" << std::endl;
            std::cout << "Logging to: " << log_filename.str() << std::endl;
        }
        
        // Setup routes
        setup_routes();
    }
    
    void start() {
        std::cout << "Starting MAE inference server on " << host_ << ":" << port_ << std::endl;
        server_.listen(host_.c_str(), port_);
    }
    
    ~MAEServer() {
        if (log_file_.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            log_file_ << "MAE Server stopped at " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << std::endl;
            log_file_.close();
        }
    }
    
private:
    torch::Tensor preprocess_image(const cv::Mat& img) {
        // Resize to 224x224 (or keep original if you want to test multi-resolution)
        cv::Mat img_resized;
        cv::resize(img, img_resized, cv::Size(224, 224));
        
        // Convert BGR to RGB
        cv::Mat img_rgb;
        cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);
        
        // Convert to float and normalize to [0, 1]
        cv::Mat img_float;
        img_rgb.convertTo(img_float, CV_32FC3, 1.0 / 255.0);
        
        // Convert to tensor
        torch::Tensor tensor = torch::from_blob(
            img_float.data, 
            {1, 224, 224, 3}, 
            torch::kFloat32
        ).clone();
        
        // Change from HWC to CHW format
        tensor = tensor.permute({0, 3, 1, 2});
        
        // Normalize with ImageNet statistics
        auto mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1});
        auto std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1});
        tensor = (tensor - mean) / std;
        
        return tensor.to(device_);
    }
    
    cv::Mat tensor_to_image(const torch::Tensor& tensor) {
        // Denormalize
        auto mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1}).to(device_);
        auto std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1}).to(device_);
        auto denormalized = tensor * std + mean;
        
        // Clamp to [0, 1]
        denormalized = denormalized.clamp(0, 1);
        
        // Convert to CPU and uint8
        denormalized = denormalized.mul(255).to(torch::kUInt8).to(torch::kCPU);
        
        // Convert to HWC
        denormalized = denormalized.squeeze(0).permute({1, 2, 0});
        
        // Create cv::Mat
        cv::Mat img(224, 224, CV_8UC3, denormalized.data_ptr());
        cv::Mat img_bgr;
        cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
        
        return img_bgr.clone();
    }
    
    cv::Mat reconstruct_image(const cv::Mat& input_img, float mask_ratio) {
        torch::NoGradGuard no_grad;
        
        // Preprocess
        auto input_tensor = preprocess_image(input_img);
        
        // Forward pass through MAE
        auto [loss, pred, mask] = model_->forward(input_tensor, mask_ratio);
        
        // Convert prediction to image
        // The prediction is patches, need to unpatchify
        auto reconstructed = model_->unpatchify(pred);
        
        // Convert to image
        return tensor_to_image(reconstructed);
    }
    
    cv::Mat create_masked_visualization(const cv::Mat& input_img, float mask_ratio) {
        // Resize image to 224x224
        cv::Mat img_resized;
        cv::resize(input_img, img_resized, cv::Size(224, 224));
        
        // Convert to RGB
        cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
        
        // Create mask pattern
        int patch_size = 16;
        int num_patches = 14; // 224/16 = 14
        int total_patches = num_patches * num_patches;
        
        // Generate random mask
        torch::manual_seed(42); // Use fixed seed for reproducibility
        auto noise = torch::rand({total_patches});
        auto ids_shuffle = torch::argsort(noise, 0, /*descending=*/false);
        auto ids_restore = torch::argsort(ids_shuffle, 0, /*descending=*/false);
        
        int len_keep = static_cast<int>(total_patches * (1 - mask_ratio));
        
        // Create binary mask: 0 is keep, 1 is remove
        auto mask = torch::ones({total_patches});
        mask.index_put_({torch::indexing::Slice(0, len_keep)}, 0);
        mask = torch::gather(mask, 0, ids_restore);
        
        // Apply mask to image visualization
        cv::Mat masked_img = img_resized.clone();
        
        for (int i = 0; i < num_patches; i++) {
            for (int j = 0; j < num_patches; j++) {
                int patch_idx = i * num_patches + j;
                if (mask[patch_idx].item().toFloat() == 1.0f) {
                    // This patch is masked - fill with gray
                    cv::rectangle(masked_img, 
                                cv::Point(j * patch_size, i * patch_size),
                                cv::Point((j + 1) * patch_size, (i + 1) * patch_size),
                                cv::Scalar(128, 128, 128), -1);
                }
            }
        }
        
        // Add grid lines for better visualization
        for (int i = 0; i <= num_patches; i++) {
            cv::line(masked_img, 
                    cv::Point(0, i * patch_size), 
                    cv::Point(224, i * patch_size), 
                    cv::Scalar(200, 200, 200), 1);
            cv::line(masked_img, 
                    cv::Point(i * patch_size, 0), 
                    cv::Point(i * patch_size, 224), 
                    cv::Scalar(200, 200, 200), 1);
        }
        
        // Convert back to BGR
        cv::cvtColor(masked_img, masked_img, cv::COLOR_RGB2BGR);
        
        return masked_img;
    }
    
    void log_request(const std::string& endpoint, const std::string& method, 
                     const std::string& client_ip, const std::string& status,
                     const std::string& details = "") {
        if (log_file_.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            log_file_ << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << " | "
                     << method << " " << endpoint << " | "
                     << "Client: " << client_ip << " | "
                     << "Status: " << status;
            
            if (!details.empty()) {
                log_file_ << " | " << details;
            }
            
            log_file_ << std::endl;
            log_file_.flush();
        }
    }
    
    void setup_routes() {
        // Health check
        server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            json response;
            response["status"] = "ok";
            response["service"] = "MAE Server";
            res.set_content(response.dump(), "application/json");
        });
        
        // Server info
        server_.Get("/info", [this](const httplib::Request&, httplib::Response& res) {
            json response;
            response["model"] = "Masked Autoencoder";
            response["device"] = device_.str();
            response["input_size"] = 224;
            res.set_content(response.dump(), "application/json");
        });
        
        // Endpoint 1: Mask image - creates a masked visualization
        server_.Post("/mask", [this](const httplib::Request& req, httplib::Response& res) {
            std::string client_ip = req.remote_addr;
            if (client_ip.empty()) client_ip = "unknown";
            
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Check for empty body
                if (req.body.empty()) {
                    res.status = 400;
                    res.set_content("Empty request body", "text/plain");
                    log_request("/mask", "POST", client_ip, "ERROR", "Empty request body");
                    return;
                }
                
                // Get mask ratio from header
                float mask_ratio = 0.75f;
                if (req.has_header("X-Mask-Ratio")) {
                    mask_ratio = std::stof(req.get_header_value("X-Mask-Ratio"));
                }
                
                // Validate mask ratio
                if (mask_ratio < 0.0f || mask_ratio > 1.0f) {
                    res.status = 400;
                    res.set_content("Invalid mask_ratio. Must be between 0.0 and 1.0", "text/plain");
                    log_request("/mask", "POST", client_ip, "ERROR", "Invalid mask_ratio");
                    return;
                }
                
                // Decode image
                std::vector<uchar> image_data(req.body.begin(), req.body.end());
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    res.status = 400;
                    res.set_content("Failed to decode image", "text/plain");
                    log_request("/mask", "POST", client_ip, "ERROR", "Failed to decode image");
                    return;
                }
                
                // Create masked visualization
                cv::Mat masked = create_masked_visualization(img, mask_ratio);
                
                // Encode result as PNG
                std::vector<uchar> buf;
                cv::imencode(".png", masked, buf);
                
                res.set_content(reinterpret_cast<const char*>(buf.data()), buf.size(), "image/png");
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                log_request("/mask", "POST", client_ip, "SUCCESS", 
                           "input=" + std::to_string(img.cols) + "x" + std::to_string(img.rows) +
                           ", mask_ratio=" + std::to_string(mask_ratio) +
                           ", time=" + std::to_string(duration.count()) + "ms");
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(e.what(), "text/plain");
                log_request("/mask", "POST", client_ip, "ERROR", 
                           std::string("Exception: ") + e.what());
            }
        });
        
        // Endpoint 2: Reconstruct image - applies MAE model to reconstruct from masked image
        server_.Post("/reconstruct", [this](const httplib::Request& req, httplib::Response& res) {
            std::string client_ip = req.remote_addr;
            if (client_ip.empty()) client_ip = "unknown";
            
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Check for empty body
                if (req.body.empty()) {
                    res.status = 400;
                    res.set_content("Empty request body", "text/plain");
                    log_request("/reconstruct", "POST", client_ip, "ERROR", "Empty request body");
                    return;
                }
                
                // Get mask ratio from header
                float mask_ratio = 0.75f;
                if (req.has_header("X-Mask-Ratio")) {
                    mask_ratio = std::stof(req.get_header_value("X-Mask-Ratio"));
                }
                
                // Validate mask ratio
                if (mask_ratio < 0.0f || mask_ratio > 1.0f) {
                    res.status = 400;
                    res.set_content("Invalid mask_ratio. Must be between 0.0 and 1.0", "text/plain");
                    log_request("/reconstruct", "POST", client_ip, "ERROR", "Invalid mask_ratio");
                    return;
                }
                
                // Decode image
                std::vector<uchar> image_data(req.body.begin(), req.body.end());
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    res.status = 400;
                    res.set_content("Failed to decode image", "text/plain");
                    log_request("/reconstruct", "POST", client_ip, "ERROR", 
                               "Failed to decode image - size=" + std::to_string(req.body.size()));
                    return;
                }
                
                // Reconstruct
                cv::Mat reconstructed = reconstruct_image(img, mask_ratio);
                
                // Encode result
                std::vector<uchar> buf;
                cv::imencode(".png", reconstructed, buf);
                
                res.set_content(reinterpret_cast<const char*>(buf.data()), buf.size(), "image/png");
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                log_request("/reconstruct", "POST", client_ip, "SUCCESS", 
                           "input=" + std::to_string(img.cols) + "x" + std::to_string(img.rows) +
                           ", mask_ratio=" + std::to_string(mask_ratio) +
                           ", time=" + std::to_string(duration.count()) + "ms");
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(e.what(), "text/plain");
                log_request("/reconstruct", "POST", client_ip, "ERROR", 
                           std::string("Exception: ") + e.what());
            }
        });
        
        // Endpoint 3: Mask and Reconstruct - does both operations in one call
        server_.Post("/mask_and_reconstruct", [this](const httplib::Request& req, httplib::Response& res) {
            std::string client_ip = req.remote_addr;
            if (client_ip.empty()) client_ip = "unknown";
            
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Check for empty body
                if (req.body.empty()) {
                    res.status = 400;
                    res.set_content("Empty request body", "text/plain");
                    log_request("/mask_and_reconstruct", "POST", client_ip, "ERROR", "Empty request body");
                    return;
                }
                
                // Get parameters from headers
                float mask_ratio = 0.75f;
                if (req.has_header("X-Mask-Ratio")) {
                    mask_ratio = std::stof(req.get_header_value("X-Mask-Ratio"));
                }
                
                std::string output_type = "reconstructed"; // default
                if (req.has_header("X-Output-Type")) {
                    output_type = req.get_header_value("X-Output-Type");
                }
                
                // Validate mask ratio
                if (mask_ratio < 0.0f || mask_ratio > 1.0f) {
                    res.status = 400;
                    res.set_content("Invalid mask_ratio. Must be between 0.0 and 1.0", "text/plain");
                    log_request("/mask_and_reconstruct", "POST", client_ip, "ERROR", "Invalid mask_ratio");
                    return;
                }
                
                // Decode image
                std::vector<uchar> image_data(req.body.begin(), req.body.end());
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    res.status = 400;
                    res.set_content("Failed to decode image", "text/plain");
                    log_request("/mask_and_reconstruct", "POST", client_ip, "ERROR", "Failed to decode image");
                    return;
                }
                
                // Perform operations based on output type
                cv::Mat result;
                
                if (output_type == "masked") {
                    // Only return masked visualization
                    result = create_masked_visualization(img, mask_ratio);
                } else if (output_type == "reconstructed") {
                    // Only return reconstruction
                    result = reconstruct_image(img, mask_ratio);
                } else if (output_type == "both") {
                    // Return both side by side
                    cv::Mat masked = create_masked_visualization(img, mask_ratio);
                    cv::Mat reconstructed = reconstruct_image(img, mask_ratio);
                    
                    // Create side-by-side image
                    result = cv::Mat(224, 448, CV_8UC3);
                    masked.copyTo(result(cv::Rect(0, 0, 224, 224)));
                    reconstructed.copyTo(result(cv::Rect(224, 0, 224, 224)));
                } else {
                    res.status = 400;
                    res.set_content("Invalid output_type. Must be 'masked', 'reconstructed', or 'both'", "text/plain");
                    log_request("/mask_and_reconstruct", "POST", client_ip, "ERROR", "Invalid output_type");
                    return;
                }
                
                // Encode result
                std::vector<uchar> buf;
                cv::imencode(".png", result, buf);
                
                res.set_content(reinterpret_cast<const char*>(buf.data()), buf.size(), "image/png");
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                log_request("/mask_and_reconstruct", "POST", client_ip, "SUCCESS", 
                           "input=" + std::to_string(img.cols) + "x" + std::to_string(img.rows) +
                           ", mask_ratio=" + std::to_string(mask_ratio) +
                           ", output_type=" + output_type +
                           ", time=" + std::to_string(duration.count()) + "ms");
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(e.what(), "text/plain");
                log_request("/mask_and_reconstruct", "POST", client_ip, "ERROR", 
                           std::string("Exception: ") + e.what());
            }
        });
    }
    
    MaskedAutoencoderViT model_;
    torch::Device device_;
    httplib::Server server_;
    std::string host_;
    int port_;
    std::ofstream log_file_;
};

int main(int argc, char* argv[]) {
    cxxopts::Options options("mae_server", "MAE Model REST API Server");
    
    options.add_options()
        ("c,checkpoint", "Path to model checkpoint file", cxxopts::value<std::string>())
        ("m,model", "Model type (mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14)", 
         cxxopts::value<std::string>()->default_value("mae_vit_base_patch16"))
        ("h,host", "Host address to bind", cxxopts::value<std::string>()->default_value("0.0.0.0"))
        ("p,port", "Port to listen on", cxxopts::value<int>()->default_value("8080"))
        ("help", "Print usage");
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help") || !result.count("checkpoint")) {
        std::cout << options.help() << std::endl;
        std::cout << "\nExample:\n";
        std::cout << "  ./mae_server --checkpoint checkpoints/model.pt --model mae_vit_base_patch16\n";
        return 0;
    }
    
    try {
        std::string checkpoint_path = result["checkpoint"].as<std::string>();
        std::string model_type = result["model"].as<std::string>();
        std::string host = result["host"].as<std::string>();
        int port = result["port"].as<int>();
        
        MAEServer server(checkpoint_path, model_type, host, port);
        server.start();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}