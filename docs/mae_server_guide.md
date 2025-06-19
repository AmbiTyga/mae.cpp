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

The server now provides only 3 endpoints, all accepting binary image data (no base64 or multipart forms).

### 1. Health Check - GET /health
Check if the server is running.

### 2. Server Info - GET /info
Get model and server information.

### 3. Mask Image - POST /mask
Creates a masked visualization of the input image.
- **Input**: Binary image data
- **Headers**: 
  - `X-Mask-Ratio`: (optional) Float between 0.0 and 1.0 (default: 0.75)
- **Output**: PNG image showing the masked visualization

### 4. Reconstruct Image - POST /reconstruct
Applies the MAE model to reconstruct an image.
- **Input**: Binary image data
- **Headers**: 
  - `X-Mask-Ratio`: (optional) Float between 0.0 and 1.0 (default: 0.75)
- **Output**: PNG image showing the reconstruction

### 5. Mask and Reconstruct - POST /mask_and_reconstruct
Performs both masking and reconstruction in one call.
- **Input**: Binary image data
- **Headers**: 
  - `X-Mask-Ratio`: (optional) Float between 0.0 and 1.0 (default: 0.75)
  - `X-Output-Type`: (optional) "masked", "reconstructed", or "both" (default: "reconstructed")
- **Output**: PNG image based on output type
  - "masked": Shows only the masked visualization
  - "reconstructed": Shows only the reconstruction
  - "both": Shows masked and reconstructed side by side

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

### 3. Mask Image

**cURL:**
```bash
# Create masked visualization with default mask ratio (0.75)
curl -X POST http://localhost:8080/mask \
  --data-binary "@image.jpg" \
  --output masked.png

# With custom mask ratio
curl -X POST http://localhost:8080/mask \
  -H "X-Mask-Ratio: 0.5" \
  --data-binary "@image.jpg" \
  --output masked_50.png
```

**wget:**
```bash
# Create masked visualization with default mask ratio
wget --post-file=image.jpg \
     -O masked.png \
     http://localhost:8080/mask

# With custom mask ratio
wget --post-file=image.jpg \
     --header="X-Mask-Ratio: 0.9" \
     -O masked_90.png \
     http://localhost:8080/mask
```

### 4. Reconstruct Image

**cURL:**
```bash
# Reconstruct with default mask ratio (0.75)
curl -X POST http://localhost:8080/reconstruct \
  --data-binary "@image.jpg" \
  --output reconstructed.png

# With custom mask ratio
curl -X POST http://localhost:8080/reconstruct \
  -H "X-Mask-Ratio: 0.5" \
  --data-binary "@image.jpg" \
  --output reconstructed_50.png
```

**wget:**
```bash
# Reconstruct with default mask ratio
wget --post-file=image.jpg \
     -O reconstructed.png \
     http://localhost:8080/reconstruct

# With custom mask ratio
wget --post-file=image.jpg \
     --header="X-Mask-Ratio: 0.9" \
     -O reconstructed_90.png \
     http://localhost:8080/reconstruct
```

### 5. Mask and Reconstruct

**cURL:**
```bash
# Get only reconstruction (default)
curl -X POST http://localhost:8080/mask_and_reconstruct \
  --data-binary "@image.jpg" \
  --output result.png

# Get only masked visualization
curl -X POST http://localhost:8080/mask_and_reconstruct \
  -H "X-Output-Type: masked" \
  --data-binary "@image.jpg" \
  --output masked.png

# Get both side by side
curl -X POST http://localhost:8080/mask_and_reconstruct \
  -H "X-Output-Type: both" \
  -H "X-Mask-Ratio: 0.75" \
  --data-binary "@image.jpg" \
  --output both.png
```

**wget:**
```bash
# Get only reconstruction (default)
wget --post-file=image.jpg \
     -O result.png \
     http://localhost:8080/mask_and_reconstruct

# Get only masked visualization
wget --post-file=image.jpg \
     --header="X-Output-Type: masked" \
     -O masked.png \
     http://localhost:8080/mask_and_reconstruct

# Get both side by side
wget --post-file=image.jpg \
     --header="X-Output-Type: both" \
     --header="X-Mask-Ratio: 0.75" \
     -O both.png \
     http://localhost:8080/mask_and_reconstruct
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

# Test all endpoints
echo "1. Creating masked visualization..."
curl -sX POST $SERVER/mask \
  -H "X-Mask-Ratio: $MASK_RATIO" \
  --data-binary "@$IMAGE_PATH" \
  --output "${BASENAME}_masked.png"

echo "2. Creating reconstruction..."
curl -sX POST $SERVER/reconstruct \
  -H "X-Mask-Ratio: $MASK_RATIO" \
  --data-binary "@$IMAGE_PATH" \
  --output "${BASENAME}_reconstructed.png"

echo "3. Creating side-by-side comparison..."
curl -sX POST $SERVER/mask_and_reconstruct \
  -H "X-Mask-Ratio: $MASK_RATIO" \
  -H "X-Output-Type: both" \
  --data-binary "@$IMAGE_PATH" \
  --output "${BASENAME}_comparison.png"

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
  curl -sX POST http://localhost:8080/mask_and_reconstruct \
    -H "X-Mask-Ratio: $ratio" \
    -H "X-Output-Type: both" \
    --data-binary "@$IMAGE" \
    --output "comparison_${ratio}.png"
done
```

## Python Client Examples

### Simple Client
```python
import requests
from pathlib import Path

def mask_image(image_path, mask_ratio=0.75, server="http://localhost:8080"):
    """Create masked visualization"""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{server}/mask",
            data=f.read(),
            headers={'X-Mask-Ratio': str(mask_ratio)}
        )
    
    if response.status_code == 200:
        output_path = Path(image_path).stem + "_masked.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved masked image to {output_path}")
    else:
        print(f"Error: {response.text}")

def reconstruct_image(image_path, mask_ratio=0.75, server="http://localhost:8080"):
    """Reconstruct image using MAE"""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{server}/reconstruct",
            data=f.read(),
            headers={'X-Mask-Ratio': str(mask_ratio)}
        )
    
    if response.status_code == 200:
        output_path = Path(image_path).stem + "_reconstructed.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved reconstructed image to {output_path}")
    else:
        print(f"Error: {response.text}")

def mask_and_reconstruct(image_path, mask_ratio=0.75, output_type="both", server="http://localhost:8080"):
    """Mask and reconstruct in one call"""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{server}/mask_and_reconstruct",
            data=f.read(),
            headers={
                'X-Mask-Ratio': str(mask_ratio),
                'X-Output-Type': output_type
            }
        )
    
    if response.status_code == 200:
        output_path = Path(image_path).stem + f"_{output_type}.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved result to {output_path}")
    else:
        print(f"Error: {response.text}")

# Usage
mask_image("photo.jpg", mask_ratio=0.75)
reconstruct_image("photo.jpg", mask_ratio=0.75)
mask_and_reconstruct("photo.jpg", mask_ratio=0.75, output_type="both")
```

### Advanced Client with All Features
```python
import requests
from pathlib import Path
import os

class MAEClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server = server_url
    
    def health_check(self):
        """Check server health"""
        return requests.get(f"{self.server}/health").json()
    
    def info(self):
        """Get server info"""
        return requests.get(f"{self.server}/info").json()
    
    def mask(self, image_path, mask_ratio=0.75):
        """Create masked visualization"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.server}/mask",
                data=f.read(),
                headers={'X-Mask-Ratio': str(mask_ratio)}
            )
        return response.content if response.status_code == 200 else None
    
    def reconstruct(self, image_path, mask_ratio=0.75):
        """Reconstruct image"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.server}/reconstruct",
                data=f.read(),
                headers={'X-Mask-Ratio': str(mask_ratio)}
            )
        return response.content if response.status_code == 200 else None
    
    def mask_and_reconstruct(self, image_path, mask_ratio=0.75, output_type="reconstructed"):
        """Mask and reconstruct with flexible output"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.server}/mask_and_reconstruct",
                data=f.read(),
                headers={
                    'X-Mask-Ratio': str(mask_ratio),
                    'X-Output-Type': output_type
                }
            )
        return response.content if response.status_code == 200 else None
    
    def process_batch(self, image_dir, mask_ratio=0.75, output_type="both"):
        """Process all images in a directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for image_file in Path(image_dir).iterdir():
            if image_file.suffix.lower() in image_extensions:
                print(f"Processing {image_file.name}...")
                
                result = self.mask_and_reconstruct(
                    str(image_file), 
                    mask_ratio=mask_ratio,
                    output_type=output_type
                )
                
                if result:
                    output_path = image_file.parent / f"{image_file.stem}_mae_{output_type}.png"
                    with open(output_path, 'wb') as f:
                        f.write(result)
                    print(f"  Saved to {output_path.name}")
                else:
                    print(f"  Failed to process {image_file.name}")

# Usage example
client = MAEClient()

# Check server status
print(client.health_check())
print(client.info())

# Process single image
masked_data = client.mask("photo.jpg", mask_ratio=0.8)
if masked_data:
    with open("masked_result.png", "wb") as f:
        f.write(masked_data)

# Process all images in directory
client.process_batch("./images", mask_ratio=0.75, output_type="both")
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

2. **Image decoding errors**
   - Ensure image file exists and is valid
   - Try different image formats (JPEG, PNG)
   - Check file permissions

3. **CUDA errors**
   - Model will automatically fall back to CPU if GPU not available
   - Check GPU memory: `nvidia-smi`

4. **Invalid mask ratio errors**
   - Ensure mask_ratio is between 0.0 and 1.0
   - Use proper header format: `X-Mask-Ratio: 0.75`

### Debug Commands

```bash
# Check server logs
tail -f mae_server_*.log

# Test with small image
convert -size 224x224 xc:red test.jpg
wget --post-file=test.jpg \
     --header="X-Mask-Ratio: 0.75" \
     -O test_result.png \
     http://localhost:8080/reconstruct

# Monitor server performance
curl http://localhost:8080/info

# Test all endpoints
for endpoint in mask reconstruct mask_and_reconstruct; do
  echo "Testing /$endpoint..."
  curl -X POST http://localhost:8080/$endpoint \
    --data-binary "@test.jpg" \
    --output "${endpoint}_result.png"
done
```

### Performance Tips

1. **Binary Format**: All endpoints now use binary format for best performance
2. **Keep Server Running**: First request loads model, subsequent requests are faster
3. **GPU Usage**: Ensure CUDA is available for best performance
4. **Use mask_and_reconstruct**: When you need both outputs, use the combined endpoint to avoid processing twice

## Example Output

When everything works correctly:
```bash
$ ./mae_process.sh cat.jpg 0.75
Processing cat.jpg with mask_ratio=0.75
1. Creating masked visualization...
2. Creating reconstruction...
3. Creating side-by-side comparison...
Done! Created:
-rw-r--r-- 1 user user  98K Jun 18 10:30 cat_masked.png
-rw-r--r-- 1 user user 145K Jun 18 10:30 cat_reconstructed.png
-rw-r--r-- 1 user user 243K Jun 18 10:30 cat_comparison.png
```

Server logs will show:
```
2025-06-18 10:30:15 | POST /mask | Client: 127.0.0.1 | Status: SUCCESS | input=512x384, mask_ratio=0.75, time=45ms
2025-06-18 10:30:16 | POST /reconstruct | Client: 127.0.0.1 | Status: SUCCESS | input=512x384, mask_ratio=0.75, time=125ms
2025-06-18 10:30:17 | POST /mask_and_reconstruct | Client: 127.0.0.1 | Status: SUCCESS | input=512x384, mask_ratio=0.75, output_type=both, time=170ms
```