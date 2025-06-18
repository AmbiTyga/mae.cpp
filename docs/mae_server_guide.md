# MAE Server Usage Guide

This guide covers building, running, and using the MAE (Masked Autoencoder) inference server for image reconstruction.

## Table of Contents
1. [Building the Server](#building-the-server)
2. [Starting the Server](#starting-the-server)
3. [API Endpoints](#api-endpoints)
4. [Using cURL with Local Images](#using-curl-with-local-images)
5. [Python Client Examples](#python-client-examples)
6. [Troubleshooting](#troubleshooting)

## Building the Server

### Prerequisites
- LibTorch (PyTorch C++)
- OpenCV
- CMake 3.14+
- C++17 compiler

### Build Steps

```bash
# Clone the repository (if not already done)
cd /path/to/mae.cpp

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Build the server
cmake --build . -j$(nproc)

# Verify the executable was created
ls -la mae_server
```

## Starting the Server

### Basic Usage

```bash
# Start with default settings (localhost:8080)
./mae_server --model /path/to/checkpoint/folder/model.pt

# Or specify the exact checkpoint file
./mae_server --model ../checkpoints/epoch-50/model.pt

# Custom host and port
./mae_server --model ../checkpoints/final.pt --host 0.0.0.0 --port 8000

# See all options
./mae_server --help
```

### Command Line Options
- `--model, -m`: Path to model checkpoint (required)
- `--host, -h`: Server host (default: 127.0.0.1)
- `--port, -p`: Server port (default: 8080)
- `--help`: Show help message

### Example Server Start
```bash
# Assuming you're in the build directory
./mae_server --model ../checkpoints/final.pt --host 0.0.0.0 --port 8080
```

You should see:
```
Loaded MAE model from ../checkpoints/final.pt
Initialized MAE server with model: ../checkpoints/final.pt
Device: CUDA
Starting MAE inference server on 0.0.0.0:8080
```

## API Endpoints

### 1. Health Check
Check if the server is running:
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok",
  "service": "MAE Inference Server"
}
```

### 2. Server Info
Get model information:
```bash
curl http://localhost:8080/info
```

Response:
```json
{
  "model": "Masked Autoencoder Vision Transformer",
  "device": "cuda:0",
  "input_size": 224
}
```

### 3. Visualize Patches
Show how an image is divided into patches:
```bash
# Endpoint: POST /visualize_patches
# Input: image (base64), show_numbers (boolean)
# Output: patched_image with grid overlay
```

### 4. Mask Image
Create a masked version of the image:
```bash
# Endpoint: POST /mask_image
# Input: image (base64), mask_ratio (float 0-1)
# Output: masked_image, mask array, statistics
```

### 5. Reconstruct Image
Reconstruct an image with masked patches:
```bash
# Endpoint: POST /reconstruct
# Input: image (base64), mask_ratio (float 0-1)
# Output: reconstructed image
```

## Using cURL with Local Images

### Helper Script for Base64 Encoding

Create a helper script `encode_image.sh`:
```bash
#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_path>"
    exit 1
fi
base64 -w 0 "$1"
```

Make it executable:
```bash
chmod +x encode_image.sh
```

### cURL Examples with Local Images

#### 1. Reconstruct an Image (75% masking)
```bash
# Method 1: Using the helper script
curl -X POST http://localhost:8080/reconstruct \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(./encode_image.sh /path/to/your/image.jpg)'",
    "mask_ratio": 0.75
  }' | jq -r '.reconstruction' | base64 -d > reconstructed.png

# Method 2: Direct base64 encoding
curl -X POST http://localhost:8080/reconstruct \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w 0 /path/to/your/image.jpg)'",
    "mask_ratio": 0.75
  }' | jq -r '.reconstruction' | base64 -d > reconstructed.png
```

#### 2. Create Masked Image
```bash
# Create and save masked version
curl -X POST http://localhost:8080/mask_image \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w 0 /path/to/your/image.jpg)'",
    "mask_ratio": 0.75
  }' | jq -r '.masked_image' | base64 -d > masked.png

# Get full response with mask array
curl -X POST http://localhost:8080/mask_image \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w 0 /path/to/your/image.jpg)'",
    "mask_ratio": 0.75
  }' > mask_response.json
```

#### 3. Visualize Patches
```bash
# Show patch grid with numbers
curl -X POST http://localhost:8080/visualize_patches \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w 0 /path/to/your/image.jpg)'",
    "show_numbers": true
  }' | jq -r '.patched_image' | base64 -d > patches.png
```

### One-Line Commands

#### Quick Reconstruction
```bash
# Reconstruct with default 75% masking
curl -sX POST http://localhost:8080/reconstruct \
  -H "Content-Type: application/json" \
  -d '{"image":"'$(base64 -w 0 image.jpg)'"}' | \
  jq -r '.reconstruction' | base64 -d > output.png
```

#### Multiple Mask Ratios
```bash
# Test different mask ratios
for ratio in 0.5 0.75 0.9; do
  curl -sX POST http://localhost:8080/reconstruct \
    -H "Content-Type: application/json" \
    -d '{"image":"'$(base64 -w 0 image.jpg)'","mask_ratio":'$ratio'}' | \
    jq -r '.reconstruction' | base64 -d > "reconstructed_${ratio}.png"
done
```

### Complete Workflow Script

Create `mae_process.sh`:
```bash
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_path> [mask_ratio]"
    exit 1
fi

IMAGE_PATH=$1
MASK_RATIO=${2:-0.75}
SERVER="http://localhost:8080"
BASENAME=$(basename "$IMAGE_PATH" .${IMAGE_PATH##*.})

echo "Processing $IMAGE_PATH with mask_ratio=$MASK_RATIO"

# Encode image
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

# 1. Visualize patches
echo "Creating patch visualization..."
curl -sX POST $SERVER/visualize_patches \
  -H "Content-Type: application/json" \
  -d '{"image":"'$IMAGE_BASE64'","show_numbers":true}' | \
  jq -r '.patched_image' | base64 -d > "${BASENAME}_patches.png"

# 2. Create masked image
echo "Creating masked image..."
MASK_RESPONSE=$(curl -sX POST $SERVER/mask_image \
  -H "Content-Type: application/json" \
  -d '{"image":"'$IMAGE_BASE64'","mask_ratio":'$MASK_RATIO'}')

echo "$MASK_RESPONSE" | jq -r '.masked_image' | base64 -d > "${BASENAME}_masked.png"
echo "Masked patches: $(echo "$MASK_RESPONSE" | jq -r '.num_masked_patches')"
echo "Visible patches: $(echo "$MASK_RESPONSE" | jq -r '.num_visible_patches')"

# 3. Reconstruct
echo "Reconstructing image..."
RECON_RESPONSE=$(curl -sX POST $SERVER/reconstruct \
  -H "Content-Type: application/json" \
  -d '{"image":"'$IMAGE_BASE64'","mask_ratio":'$MASK_RATIO'}')

echo "$RECON_RESPONSE" | jq -r '.reconstruction' | base64 -d > "${BASENAME}_reconstructed.png"
echo "Processing time: $(echo "$RECON_RESPONSE" | jq -r '.processing_time_ms')ms"

echo "Done! Created:"
echo "  - ${BASENAME}_patches.png"
echo "  - ${BASENAME}_masked.png"
echo "  - ${BASENAME}_reconstructed.png"
```

Make it executable and use:
```bash
chmod +x mae_process.sh
./mae_process.sh /path/to/image.jpg 0.75
```

## Python Client Examples

### Simple Python Script
```python
import requests
import base64
import json
import sys

def process_image(image_path, mask_ratio=0.75):
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Server URL
    server_url = "http://localhost:8080"
    
    # Reconstruct image
    response = requests.post(f"{server_url}/reconstruct", 
                           json={"image": image_base64, "mask_ratio": mask_ratio})
    
    if response.status_code == 200:
        result = response.json()
        # Decode and save result
        img_data = base64.b64decode(result['reconstruction'])
        with open('reconstructed.png', 'wb') as f:
            f.write(img_data)
        print(f"Saved reconstructed image. Processing time: {result['processing_time_ms']}ms")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mae_client.py <image_path> [mask_ratio]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mask_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
    process_image(image_path, mask_ratio)
```

### Using the Provided Client
```bash
# Full demo
python scripts/mae_client.py --server http://localhost:8080 --image photo.jpg --action demo

# Just reconstruction
python scripts/mae_client.py --server http://localhost:8080 --image photo.jpg --action reconstruct --mask-ratio 0.9
```

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check if the model path is correct
   - Ensure the port is not already in use
   - Verify LibTorch is properly installed

2. **CUDA errors**
   - Model was trained on GPU but server has no GPU: The server will automatically fall back to CPU
   - Out of memory: Reduce batch size or use CPU

3. **Image decoding errors**
   - Ensure image is properly base64 encoded
   - Check that the image file exists and is readable
   - Try with a different image format (JPEG, PNG)

4. **Slow processing**
   - First request is always slower (model loading)
   - CPU inference is much slower than GPU
   - Large images are resized to 224x224 anyway

### Debug Mode

For debugging, you can save intermediate files:
```bash
# Save base64 encoded image
base64 -w 0 image.jpg > image_base64.txt

# Create JSON request
echo '{"image":"'$(cat image_base64.txt)'","mask_ratio":0.75}' > request.json

# Send request and save response
curl -X POST http://localhost:8080/reconstruct \
  -H "Content-Type: application/json" \
  -d @request.json > response.json

# Extract and decode result
jq -r '.reconstruction' response.json | base64 -d > result.png
```

### Performance Tips

1. **Batch Processing**: Use the `/reconstruct_batch` endpoint for multiple images
2. **Keep Server Running**: First request loads the model, subsequent requests are faster
3. **Use GPU**: If available, GPU inference is much faster
4. **Local Network**: Use 127.0.0.1 instead of 0.0.0.0 for local testing

## Example Output

When everything works correctly:
```bash
$ ./mae_process.sh cat.jpg 0.75
Processing cat.jpg with mask_ratio=0.75
Creating patch visualization...
Creating masked image...
Masked patches: 147
Visible patches: 49
Reconstructing image...
Processing time: 156ms
Done! Created:
  - cat_patches.png
  - cat_masked.png
  - cat_reconstructed.png
```

The server will show:
```
[INFO] 127.0.0.1:54321 "POST /visualize_patches" 200
[INFO] 127.0.0.1:54321 "POST /mask_image" 200
[INFO] 127.0.0.1:54321 "POST /reconstruct" 200
```