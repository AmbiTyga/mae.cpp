#include "mae_inference.hpp"
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
        char_array_4[3] = char_array_3[2] & 0x3f;

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
        if (i ==4) {
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

class MAEServer {
public:
    MAEServer(const std::string& model_path, const std::string& host, int port)
        : host_(host), port_(port) {
        // Initialize MAE model
        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        model_ = std::make_unique<MAEInference>(model_path, device);
        
        std::cout << "Initialized MAE server with model: " << model_path << std::endl;
        std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
        
        // Setup routes
        setup_routes();
    }
    
    void start() {
        std::cout << "Starting MAE inference server on " << host_ << ":" << port_ << std::endl;
        server_.listen(host_.c_str(), port_);
    }
    
private:
    void setup_routes() {
        // Health check endpoint
        server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            json response;
            response["status"] = "ok";
            response["service"] = "MAE Inference Server";
            res.set_content(response.dump(), "application/json");
        });
        
        // Model info endpoint
        server_.Get("/info", [this](const httplib::Request&, httplib::Response& res) {
            json response;
            response["model"] = "Masked Autoencoder Vision Transformer";
            response["device"] = model_->get_device().str();
            response["input_size"] = 224;
            res.set_content(response.dump(), "application/json");
        });
        
        // Reconstruction endpoint with multipart support
        server_.Post("/reconstruct/multipart", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Check if multipart
                if (!req.has_file("image")) {
                    json error;
                    error["error"] = "No image file in multipart request";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Get uploaded file
                const auto& file = req.get_file_value("image");
                
                // Get mask ratio from form data
                float mask_ratio = 0.75f;
                if (req.has_param("mask_ratio")) {
                    mask_ratio = std::stof(req.get_param_value("mask_ratio"));
                }
                
                // Decode image from file content
                std::vector<uchar> image_data(file.content.begin(), file.content.end());
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    json error;
                    error["error"] = "Failed to decode image";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Run reconstruction
                cv::Mat reconstructed = model_->reconstruct_image(img, mask_ratio);
                
                // Encode result
                std::vector<uchar> buf;
                cv::imencode(".png", reconstructed, buf);
                std::string result_base64 = base64_encode(buf);
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                // Create response
                json response;
                response["reconstruction"] = result_base64;
                response["mask_ratio"] = mask_ratio;
                response["processing_time_ms"] = duration.count();
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                json error;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Reconstruction endpoint (original JSON version)
        server_.Post("/reconstruct", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Parse JSON request
                json request = json::parse(req.body);
                
                // Get image data
                std::string image_base64 = request["image"];
                float mask_ratio = request.value("mask_ratio", 0.75f);
                
                // Validate mask ratio
                if (mask_ratio < 0.0f || mask_ratio > 1.0f) {
                    json error;
                    error["error"] = "Invalid mask_ratio. Must be between 0.0 and 1.0";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Decode base64 image
                auto image_data = base64_decode(image_base64);
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    json error;
                    error["error"] = "Failed to decode image";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Run reconstruction
                cv::Mat reconstructed = model_->reconstruct_image(img, mask_ratio);
                
                // Encode result
                std::vector<uchar> buf;
                cv::imencode(".png", reconstructed, buf);
                std::string result_base64 = base64_encode(buf);
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                // Create response
                json response;
                response["reconstruction"] = result_base64;
                response["mask_ratio"] = mask_ratio;
                response["processing_time_ms"] = duration.count();
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                json error;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Binary reconstruction endpoint - returns raw image bytes
        server_.Post("/reconstruct/binary", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                // Get raw image bytes from request body
                std::vector<uchar> image_data(req.body.begin(), req.body.end());
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    res.status = 400;
                    res.set_content("Failed to decode image", "text/plain");
                    return;
                }
                
                // Get mask ratio from header or use default
                float mask_ratio = 0.75f;
                if (req.has_header("X-Mask-Ratio")) {
                    mask_ratio = std::stof(req.get_header_value("X-Mask-Ratio"));
                }
                
                // Run reconstruction
                cv::Mat reconstructed = model_->reconstruct_image(img, mask_ratio);
                
                // Encode as PNG and return binary
                std::vector<uchar> buf;
                cv::imencode(".png", reconstructed, buf);
                
                res.set_content(reinterpret_cast<const char*>(buf.data()), buf.size(), "image/png");
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(e.what(), "text/plain");
            }
        });
        
        // Mask image endpoint - creates a masked version of the image
        server_.Post("/mask_image", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                // Parse JSON request
                json request = json::parse(req.body);
                
                // Get image data
                std::string image_base64 = request["image"];
                float mask_ratio = request.value("mask_ratio", 0.75f);
                
                // Validate mask ratio
                if (mask_ratio < 0.0f || mask_ratio > 1.0f) {
                    json error;
                    error["error"] = "Invalid mask_ratio. Must be between 0.0 and 1.0";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Decode base64 image
                auto image_data = base64_decode(image_base64);
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    json error;
                    error["error"] = "Failed to decode image";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Resize image to 224x224
                cv::Mat img_resized;
                cv::resize(img, img_resized, cv::Size(224, 224));
                
                // Convert to RGB
                cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
                
                // Create mask pattern
                int patch_size = 16;
                int num_patches = 14; // 224/16 = 14
                int total_patches = num_patches * num_patches;
                
                // Generate random mask
                torch::manual_seed(42); // Use fixed seed for reproducibility or random
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
                
                // Encode result
                std::vector<uchar> buf;
                cv::imencode(".png", masked_img, buf);
                std::string masked_base64 = base64_encode(buf);
                
                // Convert mask to 2D array for response
                std::vector<std::vector<float>> mask_array(num_patches, std::vector<float>(num_patches));
                for (int i = 0; i < num_patches; i++) {
                    for (int j = 0; j < num_patches; j++) {
                        mask_array[i][j] = 1.0f - mask[i * num_patches + j].item().toFloat();
                    }
                }
                
                // Create response
                json response;
                response["masked_image"] = masked_base64;
                response["mask"] = mask_array;
                response["mask_ratio"] = mask_ratio;
                response["num_masked_patches"] = total_patches - len_keep;
                response["num_visible_patches"] = len_keep;
                response["patch_size"] = patch_size;
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                json error;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Visualize patches endpoint - shows how the image is divided into patches
        server_.Post("/visualize_patches", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                // Parse JSON request
                json request = json::parse(req.body);
                
                // Get image data
                std::string image_base64 = request["image"];
                bool show_numbers = request.value("show_numbers", true);
                
                // Decode base64 image
                auto image_data = base64_decode(image_base64);
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    json error;
                    error["error"] = "Failed to decode image";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Resize image to 224x224
                cv::Mat img_resized;
                cv::resize(img, img_resized, cv::Size(224, 224));
                
                // Draw patch grid
                int patch_size = 16;
                int num_patches = 14; // 224/16 = 14
                
                // Draw grid lines
                for (int i = 0; i <= num_patches; i++) {
                    cv::line(img_resized, 
                            cv::Point(0, i * patch_size), 
                            cv::Point(224, i * patch_size), 
                            cv::Scalar(0, 255, 0), 2);
                    cv::line(img_resized, 
                            cv::Point(i * patch_size, 0), 
                            cv::Point(i * patch_size, 224), 
                            cv::Scalar(0, 255, 0), 2);
                }
                
                // Optionally add patch numbers
                if (show_numbers) {
                    for (int i = 0; i < num_patches; i++) {
                        for (int j = 0; j < num_patches; j++) {
                            int patch_idx = i * num_patches + j;
                            cv::putText(img_resized, 
                                      std::to_string(patch_idx),
                                      cv::Point(j * patch_size + 2, i * patch_size + 14),
                                      cv::FONT_HERSHEY_SIMPLEX, 
                                      0.3, 
                                      cv::Scalar(255, 255, 255), 
                                      1);
                        }
                    }
                }
                
                // Encode result
                std::vector<uchar> buf;
                cv::imencode(".png", img_resized, buf);
                std::string result_base64 = base64_encode(buf);
                
                // Create response
                json response;
                response["patched_image"] = result_base64;
                response["patch_size"] = patch_size;
                response["num_patches_per_side"] = num_patches;
                response["total_patches"] = num_patches * num_patches;
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                json error;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Multi-resolution reconstruction endpoint - NEW!
        server_.Post("/reconstruct_multisize", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Parse JSON request
                json request = json::parse(req.body);
                
                // Get image data
                std::string image_base64 = request["image"];
                float mask_ratio = request.value("mask_ratio", 0.75f);
                int target_size = request.value("size", 224);  // Default 224, can be 448, 672, etc.
                
                // Validate size (must be divisible by patch size 16)
                if (target_size % 16 != 0) {
                    json error;
                    error["error"] = "Size must be divisible by 16 (patch size)";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Decode base64 image
                auto image_data = base64_decode(image_base64);
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                
                if (img.empty()) {
                    json error;
                    error["error"] = "Failed to decode image";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Resize to target size
                cv::Mat img_resized;
                cv::resize(img, img_resized, cv::Size(target_size, target_size));
                
                // Run reconstruction at the new size
                cv::Mat reconstructed = model_->reconstruct_image(img_resized, mask_ratio);
                
                // Encode result
                std::vector<uchar> buf;
                cv::imencode(".png", reconstructed, buf);
                std::string result_base64 = base64_encode(buf);
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                // Create response
                json response;
                response["reconstruction"] = result_base64;
                response["mask_ratio"] = mask_ratio;
                response["input_size"] = target_size;
                response["num_patches"] = (target_size / 16) * (target_size / 16);
                response["processing_time_ms"] = duration.count();
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                json error;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Batch reconstruction endpoint
        server_.Post("/reconstruct_batch", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Parse JSON request
                json request = json::parse(req.body);
                
                // Get images array
                auto images_array = request["images"];
                float mask_ratio = request.value("mask_ratio", 0.75f);
                
                if (!images_array.is_array()) {
                    json error;
                    error["error"] = "Images must be an array";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Decode all images
                std::vector<cv::Mat> images;
                for (const auto& img_base64 : images_array) {
                    auto image_data = base64_decode(img_base64.get<std::string>());
                    cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                    if (!img.empty()) {
                        images.push_back(img);
                    }
                }
                
                if (images.empty()) {
                    json error;
                    error["error"] = "No valid images provided";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                // Process batch
                model_->set_mask_ratio(mask_ratio);
                auto outputs = model_->infer_batch(images);
                
                // Encode results
                json response;
                response["reconstructions"] = json::array();
                
                for (int i = 0; i < outputs.size(0); ++i) {
                    auto reconstruction = model_->get_reconstruction(outputs[i]);
                    std::vector<uchar> buf;
                    cv::imencode(".png", reconstruction, buf);
                    response["reconstructions"].push_back(base64_encode(buf));
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                response["batch_size"] = images.size();
                response["mask_ratio"] = mask_ratio;
                response["processing_time_ms"] = duration.count();
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                json error;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
    }
    
    std::unique_ptr<MAEInference> model_;
    httplib::Server server_;
    std::string host_;
    int port_;
};

int main(int argc, char* argv[]) {
    cxxopts::Options options("mae_server", "MAE Inference REST API Server");
    
    options.add_options()
        ("m,model", "Path to TorchScript model file", cxxopts::value<std::string>())
        ("h,host", "Host address to bind", cxxopts::value<std::string>()->default_value("0.0.0.0"))
        ("p,port", "Port to listen on", cxxopts::value<int>()->default_value("8080"))
        ("help", "Print usage");
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help") || !result.count("model")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    
    try {
        std::string model_path = result["model"].as<std::string>();
        std::string host = result["host"].as<std::string>();
        int port = result["port"].as<int>();
        
        MAEServer server(model_path, host, port);
        server.start();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}