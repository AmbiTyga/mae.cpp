#include "mae_model.hpp"
#include <httplib.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <base64.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cxxopts.hpp>
#include <iomanip>

using json = nlohmann::json;

// Base64 encoding/decoding utilities
static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uchar>& buf) {
    std::string ret;
    int i = 0;
    int j = 0;
    uchar char_array_3[3];
    uchar char_array_4[4];
    size_t buflen = buf.size();
    const uchar* data = buf.data();

    while (buflen--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i < 4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }

    return ret;
}

std::vector<uchar> base64_decode(const std::string& encoded_string) {
    size_t in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    uchar char_array_4[4], char_array_3[3];
    std::vector<uchar> ret;

    while (in_len-- && (encoded_string[in_] != '=') && 
           (isalnum(encoded_string[in_]) || (encoded_string[in_] == '+') || (encoded_string[in_] == '/'))) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]) & 0xff;

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = 0; j < i; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]) & 0xff;

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        
        for (j = 0; (j < i - 1); j++) 
            ret.push_back(char_array_3[j]);
    }

    return ret;
}

class MAENativeServer {
public:
    MAENativeServer(const std::string& checkpoint_path, const std::string& model_type, 
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
        log_filename << "mae_native_server_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
        log_file_.open(log_filename.str(), std::ios::app);
        
        if (log_file_.is_open()) {
            log_file_ << "MAE Native Server started at " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << std::endl;
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
        std::cout << "Starting MAE Native inference server on " << host_ << ":" << port_ << std::endl;
        server_.listen(host_.c_str(), port_);
    }
    
    ~MAENativeServer() {
        if (log_file_.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            log_file_ << "MAE Native Server stopped at " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << std::endl;
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
            response["service"] = "MAE Native Server";
            res.set_content(response.dump(), "application/json");
        });
        
        // Server info
        server_.Get("/info", [this](const httplib::Request&, httplib::Response& res) {
            json response;
            response["model"] = "Masked Autoencoder (Native)";
            response["device"] = device_.str();
            response["input_size"] = 224;
            res.set_content(response.dump(), "application/json");
        });
        
        // Binary reconstruction endpoint
        server_.Post("/reconstruct/binary", [this](const httplib::Request& req, httplib::Response& res) {
            std::string client_ip = req.remote_addr;
            if (client_ip.empty()) client_ip = "unknown";
            
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Check for empty body
                if (req.body.empty()) {
                    res.status = 400;
                    res.set_content("Empty request body", "text/plain");
                    log_request("/reconstruct/binary", "POST", client_ip, "ERROR", "Empty request body");
                    return;
                }
                
                // Get image data
                std::vector<uchar> image_data(req.body.begin(), req.body.end());
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    res.status = 400;
                    res.set_content("Failed to decode image", "text/plain");
                    log_request("/reconstruct/binary", "POST", client_ip, "ERROR", 
                               "Failed to decode image - size=" + std::to_string(req.body.size()));
                    return;
                }
                
                // Get mask ratio
                float mask_ratio = 0.75f;
                if (req.has_header("X-Mask-Ratio")) {
                    mask_ratio = std::stof(req.get_header_value("X-Mask-Ratio"));
                }
                
                // Reconstruct
                cv::Mat reconstructed = reconstruct_image(img, mask_ratio);
                
                // Encode result
                std::vector<uchar> buf;
                cv::imencode(".png", reconstructed, buf);
                
                res.set_content(reinterpret_cast<const char*>(buf.data()), buf.size(), "image/png");
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                log_request("/reconstruct/binary", "POST", client_ip, "SUCCESS", 
                           "input=" + std::to_string(img.cols) + "x" + std::to_string(img.rows) +
                           ", mask_ratio=" + std::to_string(mask_ratio) +
                           ", time=" + std::to_string(duration.count()) + "ms");
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(e.what(), "text/plain");
                log_request("/reconstruct/binary", "POST", client_ip, "ERROR", 
                           std::string("Exception: ") + e.what());
            }
        });
        
        // Multipart reconstruction endpoint
        server_.Post("/reconstruct/multipart", [this](const httplib::Request& req, httplib::Response& res) {
            std::string client_ip = req.remote_addr;
            if (client_ip.empty()) client_ip = "unknown";
            
            try {
                if (!req.has_file("image")) {
                    json error;
                    error["error"] = "No image file in multipart request";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    log_request("/reconstruct/multipart", "POST", client_ip, "ERROR", "No image file");
                    return;
                }
                
                const auto& file = req.get_file_value("image");
                std::vector<uchar> image_data(file.content.begin(), file.content.end());
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    json error;
                    error["error"] = "Failed to decode image";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    log_request("/reconstruct/multipart", "POST", client_ip, "ERROR", 
                               "Failed to decode image - " + file.filename);
                    return;
                }
                
                float mask_ratio = 0.75f;
                if (req.has_param("mask_ratio")) {
                    mask_ratio = std::stof(req.get_param_value("mask_ratio"));
                }
                
                cv::Mat reconstructed = reconstruct_image(img, mask_ratio);
                
                std::vector<uchar> buf;
                cv::imencode(".png", reconstructed, buf);
                std::string result_base64 = base64_encode(buf);
                
                json response;
                response["reconstruction"] = result_base64;
                response["mask_ratio"] = mask_ratio;
                
                res.set_content(response.dump(), "application/json");
                
                log_request("/reconstruct/multipart", "POST", client_ip, "SUCCESS", 
                           file.filename + ", mask_ratio=" + std::to_string(mask_ratio));
                
            } catch (const std::exception& e) {
                json error;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
                log_request("/reconstruct/multipart", "POST", client_ip, "ERROR", 
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
    cxxopts::Options options("mae_native_server", "MAE Native Model REST API Server");
    
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
        std::cout << "  ./mae_native_server --checkpoint checkpoints/model.pt --model mae_vit_base_patch16\n";
        return 0;
    }
    
    try {
        std::string checkpoint_path = result["checkpoint"].as<std::string>();
        std::string model_type = result["model"].as<std::string>();
        std::string host = result["host"].as<std::string>();
        int port = result["port"].as<int>();
        
        MAENativeServer server(checkpoint_path, model_type, host, port);
        server.start();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}