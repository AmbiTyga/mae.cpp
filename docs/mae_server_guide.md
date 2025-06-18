# MAE Server Usage Guide

This guide covers building, running, and using the MAE (Masked Autoencoder) inference server for image reconstruction.

## Table of Contents
1. [Building the Server](#building-the-server)
2. [Starting the Server](#starting-the-server)
3. [API Endpoints](#api-endpoints)
4. [Complete Usage Examples](#complete-usage-examples)
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
cd /path/to/MAE

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
# Start with a native PyTorch checkpoint
./mae_server --checkpoint /path/to/checkpoint.pt --model mae_vit_base_patch16

# Custom host and port
./mae_server --checkpoint ../checkpoints/model.pt --model mae_vit_base_patch16 --host 0.0.0.0 --port 8000

# See all options
./mae_server --help
```

### Command Line Options
- `--checkpoint, -c`: Path to model checkpoint file (required)
- `--model, -m`: Model type (mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14)
- `--host, -h`: Server host (default: 0.0.0.0)
- `--port, -p`: Server port (default: 8080)
- `--help`: Show help message

### Example Server Start
```bash
# Assuming you're in the build directory
./mae_server --checkpoint ../checkpoints/step-23000/model.pt --model mae_vit_base_patch16
```

You should see:
```
Loaded MAE model from: ../checkpoints/step-23000/model.pt
Model type: mae_vit_base_patch16
Device: CUDA
Starting MAE inference server on 0.0.0.0:8080
```

## API Endpoints

### 1. Health Check - GET /health
Check if the server is running.

### 2. Server Info - GET /info
Get model and server information.

### 3. Binary Reconstruction - POST /reconstruct/binary
Send raw image bytes, receive PNG image.

### 4. Multipart Reconstruction - POST /reconstruct/multipart
Send image as multipart form data.

### 5. JSON Reconstruction - POST /reconstruct
Send base64 encoded image in JSON.

### 6. Mask Image
- Binary: POST /mask_image/binary - Send raw image bytes, returns PNG image
- Multipart: POST /mask_image/multipart - Send as form data, returns JSON with base64
- JSON: POST /mask_image - Base64 encoded image (for compatibility)

### 7. Visualize Patches
- Binary: POST /visualize_patches/binary - Send raw image bytes, returns PNG image
- Multipart: POST /visualize_patches/multipart - Send as form data, returns JSON with base64
- JSON: POST /visualize_patches - Base64 encoded image (for compatibility)

### 8. Multi-resolution Reconstruction
- Binary: POST /reconstruct_multisize/binary - Send raw image bytes with size in header
- Multipart: POST /reconstruct_multisize/multipart - Send as form data with size parameter

### 9. Batch Reconstruction - POST /reconstruct_batch
Process multiple images at once. (Note: This endpoint only supports JSON/base64 format due to the nature of batch processing)

## Complete Usage Examples

### 1. Health Check

**cURL:**
```bash
curl http://localhost:8080/health
```

**wget:**
```bash
wget -qO- http://localhost:8080/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "MAE Server"
}
```

### 2. Server Info

**cURL:**
```bash
curl http://localhost:8080/info
```

**wget:**
```bash
wget -qO- http://localhost:8080/info
```

**Response:**
```json
{
  "model": "Masked Autoencoder",
  "device": "cuda:0",
  "input_size": 224
}
```

### 3. Binary Reconstruction (Recommended)

**cURL:**
```bash
# Basic reconstruction
curl -X POST http://localhost:8080/reconstruct/binary \
  -H "Content-Type: image/jpeg" \
  -H "X-Mask-Ratio: 0.75" \
  --data-binary "@image.jpg" \
  --output reconstructed.png

# Different mask ratios
curl -X POST http://localhost:8080/reconstruct/binary \
  -H "Content-Type: image/jpeg" \
  -H "X-Mask-Ratio: 0.5" \
  --data-binary "@image.jpg" \
  --output reconstructed_50.png
```

**wget:**
```bash
# Basic reconstruction
wget --post-file=image.jpg \
     --header="Content-Type: image/jpeg" \
     --header="X-Mask-Ratio: 0.75" \
     -O reconstructed.png \
     http://localhost:8080/reconstruct/binary

# Different mask ratios
wget --post-file=image.jpg \
     --header="X-Mask-Ratio: 0.9" \
     -O reconstructed_90.png \
     http://localhost:8080/reconstruct/binary
```

### 4. Multipart Form Data

**cURL:**
```bash
# Send as multipart
curl -X POST http://localhost:8080/reconstruct/multipart \
  -F "image=@image.jpg" \
  -F "mask_ratio=0.75" \
  --output response.json

# Extract base64 result
curl -X POST http://localhost:8080/reconstruct/multipart \
  -F "image=@image.jpg" \
  -F "mask_ratio=0.75" | \
  jq -r '.reconstruction' | base64 -d > reconstructed.png
```

**wget (using a helper script):**
```bash
# Create multipart form data file
cat > form_data.txt << 'EOF'
--boundary
Content-Disposition: form-data; name="image"; filename="image.jpg"
Content-Type: image/jpeg

$(cat image.jpg | xxd -p | tr -d '\n' | xxd -r -p)
--boundary
Content-Disposition: form-data; name="mask_ratio"

0.75
--boundary--
EOF

# Send request
wget --post-file=form_data.txt \
     --header="Content-Type: multipart/form-data; boundary=boundary" \
     -O response.json \
     http://localhost:8080/reconstruct/multipart
```

### 5. JSON with Base64 Encoding

**cURL:**
```bash
# One-liner
curl -X POST http://localhost:8080/reconstruct \
  -H "Content-Type: application/json" \
  -d '{"image":"'$(base64 -w 0 image.jpg)'","mask_ratio":0.75}' | \
  jq -r '.reconstruction' | base64 -d > reconstructed.png

# With intermediate steps
IMAGE_BASE64=$(base64 -w 0 image.jpg)
curl -X POST http://localhost:8080/reconstruct \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$IMAGE_BASE64\",\"mask_ratio\":0.75}" | \
  jq -r '.reconstruction' | base64 -d > reconstructed.png
```

**wget:**
```bash
# Create JSON request
echo '{"image":"'$(base64 -w 0 image.jpg)'","mask_ratio":0.75}' > request.json

# Send request
wget -qO- --post-data="@request.json" \
     --header="Content-Type: application/json" \
     http://localhost:8080/reconstruct | \
     jq -r '.reconstruction' | base64 -d > reconstructed.png
```

### 6. Mask Image Visualization

#### Binary Endpoint (Recommended)

**cURL:**
```bash
# Create masked visualization - returns PNG directly
curl -X POST http://localhost:8080/mask_image/binary \
  -H "X-Mask-Ratio: 0.75" \
  --data-binary "@image.jpg" \
  --output masked.png

# Different mask ratios
curl -X POST http://localhost:8080/mask_image/binary \
  -H "X-Mask-Ratio: 0.5" \
  --data-binary "@image.jpg" \
  --output masked_50.png
```

**wget:**
```bash
# Create masked visualization
wget --post-file=image.jpg \
     --header="X-Mask-Ratio: 0.75" \
     -O masked.png \
     http://localhost:8080/mask_image/binary

# Different mask ratios
wget --post-file=image.jpg \
     --header="X-Mask-Ratio: 0.9" \
     -O masked_90.png \
     http://localhost:8080/mask_image/binary
```

#### Multipart Endpoint

**cURL:**
```bash
# Create masked visualization with multipart
curl -X POST http://localhost:8080/mask_image/multipart \
  -F "image=@image.jpg" \
  -F "mask_ratio=0.75" \
  --output mask_response.json

# Extract the masked image
curl -X POST http://localhost:8080/mask_image/multipart \
  -F "image=@image.jpg" \
  -F "mask_ratio=0.75" | \
  jq -r '.masked_image' | base64 -d > masked.png

# Get mask statistics
curl -X POST http://localhost:8080/mask_image/multipart \
  -F "image=@image.jpg" \
  -F "mask_ratio=0.75" | \
  jq '.num_masked_patches, .num_visible_patches'
```

**wget:**
```bash
# Create form data
cat > mask_form.txt << 'EOF'
--boundary
Content-Disposition: form-data; name="image"; filename="image.jpg"
Content-Type: image/jpeg

$(cat image.jpg)
--boundary
Content-Disposition: form-data; name="mask_ratio"

0.75
--boundary--
EOF

# Send request
wget --post-file=mask_form.txt \
     --header="Content-Type: multipart/form-data; boundary=boundary" \
     -O mask_response.json \
     http://localhost:8080/mask_image/multipart
```

#### JSON Endpoint (Legacy/Compatibility)

**cURL:**
```bash
# For compatibility with existing code
curl -X POST http://localhost:8080/mask_image \
  -H "Content-Type: application/json" \
  -d '{"image":"'$(base64 -w 0 image.jpg)'","mask_ratio":0.75}' | \
  jq -r '.masked_image' | base64 -d > masked.png
```

### 7. Visualize Patches

#### Binary Endpoint (Recommended)

**cURL:**
```bash
# Show patch grid with numbers - returns PNG directly
curl -X POST http://localhost:8080/visualize_patches/binary \
  -H "X-Show-Numbers: true" \
  --data-binary "@image.jpg" \
  --output patches.png

# Without numbers
curl -X POST http://localhost:8080/visualize_patches/binary \
  -H "X-Show-Numbers: false" \
  --data-binary "@image.jpg" \
  --output patches_plain.png
```

**wget:**
```bash
# Show patch grid with numbers
wget --post-file=image.jpg \
     --header="X-Show-Numbers: true" \
     -O patches.png \
     http://localhost:8080/visualize_patches/binary

# Without numbers
wget --post-file=image.jpg \
     --header="X-Show-Numbers: false" \
     -O patches_plain.png \
     http://localhost:8080/visualize_patches/binary
```

#### Multipart Endpoint

**cURL:**
```bash
# Show patch grid with multipart
curl -X POST http://localhost:8080/visualize_patches/multipart \
  -F "image=@image.jpg" \
  -F "show_numbers=true" \
  --output patches_response.json

# Extract the image
curl -X POST http://localhost:8080/visualize_patches/multipart \
  -F "image=@image.jpg" \
  -F "show_numbers=true" | \
  jq -r '.patched_image' | base64 -d > patches.png
```

**wget:**
```bash
# Create form data
cat > patches_form.txt << 'EOF'
--boundary
Content-Disposition: form-data; name="image"; filename="image.jpg"
Content-Type: image/jpeg

$(cat image.jpg)
--boundary
Content-Disposition: form-data; name="show_numbers"

true
--boundary--
EOF

# Send request
wget --post-file=patches_form.txt \
     --header="Content-Type: multipart/form-data; boundary=boundary" \
     -O patches_response.json \
     http://localhost:8080/visualize_patches/multipart
```

#### JSON Endpoint (Legacy/Compatibility)

**cURL:**
```bash
# For compatibility with existing code
curl -X POST http://localhost:8080/visualize_patches \
  -H "Content-Type: application/json" \
  -d '{"image":"'$(base64 -w 0 image.jpg)'","show_numbers":true}' | \
  jq -r '.patched_image' | base64 -d > patches.png
```

### 8. Multi-resolution Reconstruction

#### Binary Endpoint (Recommended)

**cURL:**
```bash
# Reconstruct at 448x448
curl -X POST http://localhost:8080/reconstruct_multisize/binary \
  -H "X-Mask-Ratio: 0.75" \
  -H "X-Target-Size: 448" \
  --data-binary "@image.jpg" \
  --output reconstructed_448.png

# Reconstruct at 672x672
curl -X POST http://localhost:8080/reconstruct_multisize/binary \
  -H "X-Mask-Ratio: 0.75" \
  -H "X-Target-Size: 672" \
  --data-binary "@image.jpg" \
  --output reconstructed_672.png

# Test multiple sizes
for size in 224 448 672; do
  curl -X POST http://localhost:8080/reconstruct_multisize/binary \
    -H "X-Mask-Ratio: 0.75" \
    -H "X-Target-Size: $size" \
    --data-binary "@image.jpg" \
    --output "reconstructed_${size}.png"
done
```

**wget:**
```bash
# Reconstruct at 448x448
wget --post-file=image.jpg \
     --header="X-Mask-Ratio: 0.75" \
     --header="X-Target-Size: 448" \
     -O reconstructed_448.png \
     http://localhost:8080/reconstruct_multisize/binary

# Reconstruct at 672x672
wget --post-file=image.jpg \
     --header="X-Mask-Ratio: 0.75" \
     --header="X-Target-Size: 672" \
     -O reconstructed_672.png \
     http://localhost:8080/reconstruct_multisize/binary
```

#### Multipart Endpoint

**cURL:**
```bash
# Reconstruct at specific size with multipart
curl -X POST http://localhost:8080/reconstruct_multisize/multipart \
  -F "image=@image.jpg" \
  -F "mask_ratio=0.75" \
  -F "size=448" \
  --output response.json

# Extract result
curl -X POST http://localhost:8080/reconstruct_multisize/multipart \
  -F "image=@image.jpg" \
  -F "mask_ratio=0.75" \
  -F "size=448" | \
  jq -r '.reconstruction' | base64 -d > reconstructed_448.png
```

**wget:**
```bash
# Using wget with multipart (requires form file creation)
# Create form data
cat > multisize_form.txt << 'EOF'
--boundary
Content-Disposition: form-data; name="image"; filename="image.jpg"
Content-Type: image/jpeg

$(cat image.jpg)
--boundary
Content-Disposition: form-data; name="mask_ratio"

0.75
--boundary
Content-Disposition: form-data; name="size"

448
--boundary--
EOF

# Send request
wget --post-file=multisize_form.txt \
     --header="Content-Type: multipart/form-data; boundary=boundary" \
     -O response.json \
     http://localhost:8080/reconstruct_multisize/multipart
```

### 9. Batch Processing

**cURL:**
```bash
# Process multiple images
curl -X POST http://localhost:8080/reconstruct_batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "'$(base64 -w 0 image1.jpg)'",
      "'$(base64 -w 0 image2.jpg)'",
      "'$(base64 -w 0 image3.jpg)'"
    ],
    "mask_ratio": 0.75
  }' > batch_response.json

# Extract all reconstructions
jq -r '.reconstructions[]' batch_response.json | while IFS= read -r img; do
  echo "$img" | base64 -d > "batch_output_$((++i)).png"
done
```

**wget:**
```bash
# Create batch request
cat > batch_request.json << EOF
{
  "images": [
    "$(base64 -w 0 image1.jpg)",
    "$(base64 -w 0 image2.jpg)"
  ],
  "mask_ratio": 0.75
}
EOF

# Send batch request
wget -qO- --post-file=batch_request.json \
     --header="Content-Type: application/json" \
     http://localhost:8080/reconstruct_batch > batch_response.json
```

## Utility Scripts

### Complete Processing Script

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

# Test different endpoints
echo "1. Binary endpoint (fastest)..."
curl -sX POST $SERVER/reconstruct/binary \
  -H "X-Mask-Ratio: $MASK_RATIO" \
  --data-binary "@$IMAGE_PATH" \
  --output "${BASENAME}_binary.png"

echo "2. Creating masked visualization..."
curl -sX POST $SERVER/mask_image/binary \
  -H "X-Mask-Ratio: $MASK_RATIO" \
  --data-binary "@$IMAGE_PATH" \
  --output "${BASENAME}_masked.png"

echo "3. Creating patch visualization..."
curl -sX POST $SERVER/visualize_patches/binary \
  -H "X-Show-Numbers: true" \
  --data-binary "@$IMAGE_PATH" \
  --output "${BASENAME}_patches.png"

echo "Done! Created:"
ls -la "${BASENAME}"_*.png
```

### Batch Testing Script

Create `test_mask_ratios.sh`:
```bash
#!/bin/bash

IMAGE=$1
for ratio in 0.25 0.5 0.75 0.9; do
  echo "Testing mask_ratio=$ratio"
  curl -sX POST http://localhost:8080/reconstruct/binary \
    -H "X-Mask-Ratio: $ratio" \
    --data-binary "@$IMAGE" \
    --output "reconstructed_${ratio}.png"
done
```

## Python Client Examples

### Simple Client
```python
import requests
import base64
from pathlib import Path

def reconstruct_image(image_path, mask_ratio=0.75, server="http://localhost:8080"):
    # Method 1: Binary endpoint (recommended)
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{server}/reconstruct/binary",
            data=f.read(),
            headers={
                'Content-Type': 'image/jpeg',
                'X-Mask-Ratio': str(mask_ratio)
            }
        )
    
    if response.status_code == 200:
        output_path = Path(image_path).stem + "_reconstructed.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {output_path}")
    else:
        print(f"Error: {response.text}")

# Usage
reconstruct_image("image.jpg", mask_ratio=0.75)
```

### Advanced Client with All Endpoints
```python
import requests
import base64
import json
from pathlib import Path

class MAEClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server = server_url
    
    def health_check(self):
        return requests.get(f"{self.server}/health").json()
    
    def reconstruct_binary(self, image_path, mask_ratio=0.75):
        """Fastest method - recommended"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.server}/reconstruct/binary",
                data=f.read(),
                headers={
                    'Content-Type': 'image/jpeg',
                    'X-Mask-Ratio': str(mask_ratio)
                }
            )
        return response.content if response.status_code == 200 else None
    
    def reconstruct_json(self, image_path, mask_ratio=0.75):
        """JSON method with base64"""
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            f"{self.server}/reconstruct",
            json={"image": image_base64, "mask_ratio": mask_ratio}
        )
        
        if response.status_code == 200:
            result = response.json()
            return base64.b64decode(result['reconstruction'])
        return None
    
    def mask_image(self, image_path, mask_ratio=0.75):
        """Get masked visualization"""
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            f"{self.server}/mask_image",
            json={"image": image_base64, "mask_ratio": mask_ratio}
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'image': base64.b64decode(result['masked_image']),
                'mask': result['mask'],
                'stats': {
                    'masked': result['num_masked_patches'],
                    'visible': result['num_visible_patches']
                }
            }
        return None

# Usage
client = MAEClient()
print(client.health_check())

# Reconstruct
img_data = client.reconstruct_binary("photo.jpg", mask_ratio=0.8)
with open("result.png", "wb") as f:
    f.write(img_data)
```

## Troubleshooting

### Common Issues

1. **Server won't start**
   ```bash
   # Check if port is in use
   lsof -i :8080
   
   # Check model file exists
   ls -la ../checkpoints/model.pt
   ```

2. **wget 500 errors**
   ```bash
   # Use curl instead
   curl -X POST http://localhost:8080/reconstruct/binary \
     --data-binary "@image.jpg" \
     -H "X-Mask-Ratio: 0.75" \
     --output result.png
   
   # Or check server logs
   tail -f mae_server_*.log
   ```

3. **CUDA errors**
   - Model will automatically fall back to CPU if GPU not available
   - Check GPU memory: `nvidia-smi`

4. **Image decoding errors**
   - Ensure image file exists and is valid
   - Try different image formats (JPEG, PNG)
   - Check file permissions

### Debug Commands

```bash
# Check server logs
tail -f mae_server_*.log

# Test with small image
convert -size 224x224 xc:red test.jpg
wget --post-file=test.jpg \
     --header="X-Mask-Ratio: 0.75" \
     -O test_result.png \
     http://localhost:8080/reconstruct/binary

# Monitor server performance
curl http://localhost:8080/info
```

### Performance Tips

1. **Use Binary Endpoint**: Much faster than base64 encoding
2. **Batch Processing**: Use `/reconstruct_batch` for multiple images
3. **Keep Server Running**: First request loads model, subsequent requests are faster
4. **GPU Usage**: Ensure CUDA is available for best performance

## Example Output

When everything works correctly:
```bash
$ ./mae_process.sh cat.jpg 0.75
Processing cat.jpg with mask_ratio=0.75
1. Binary endpoint (fastest)...
2. Creating masked visualization...
3. Creating patch visualization...
Done! Created:
-rw-r--r-- 1 user user 145K Jun 18 10:30 cat_binary.png
-rw-r--r-- 1 user user  98K Jun 18 10:30 cat_masked.png
-rw-r--r-- 1 user user 134K Jun 18 10:30 cat_patches.png
```

Server logs will show:
```
2025-06-18 10:30:15 | POST /reconstruct/binary | Client: 127.0.0.1 | Status: SUCCESS | input=512x384, mask_ratio=0.75, time=125ms
2025-06-18 10:30:16 | POST /mask_image | Client: 127.0.0.1 | Status: SUCCESS | mask_ratio=0.75
2025-06-18 10:30:17 | POST /visualize_patches | Client: 127.0.0.1 | Status: SUCCESS | show_numbers=1
```