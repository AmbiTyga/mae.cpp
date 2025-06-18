# MAE Inference Server

This document describes how to use the MAE C++ inference server for serving Masked Autoencoder models via REST API.

## Building the Server

After training your MAE model, you can build the inference server:

```bash
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . -j$(nproc)
```

This will create the `mae_server` executable.

## Exporting Model for Inference

The training script automatically exports a TorchScript model at the end of training:
- Location: `checkpoints/mae_model.pt`

You can also manually export a checkpoint:

```python
import torch
from mae_model import mae_vit_base_patch16_dec512d8b

# Load checkpoint
checkpoint = torch.load('checkpoints/mae_epoch_399.pt')
model = mae_vit_base_patch16_dec512d8b()
model.load_state_dict(checkpoint['model'])
model.eval()

# Export as TorchScript
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, (example_input, 0.75))
traced_model.save('mae_model.pt')
```

## Running the Server

```bash
./mae_server --model checkpoints/mae_model.pt --host 0.0.0.0 --port 8080
```

Options:
- `--model, -m`: Path to TorchScript model file (required)
- `--host, -h`: Host address to bind (default: 0.0.0.0)
- `--port, -p`: Port to listen on (default: 8080)

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "service": "MAE Inference Server"
}
```

### Model Information
```bash
GET /info
```

Response:
```json
{
  "model": "Masked Autoencoder Vision Transformer",
  "device": "cuda",
  "input_size": 224
}
```

### Single Image Reconstruction
```bash
POST /reconstruct
```

Request:
```json
{
  "image": "<base64_encoded_image>",
  "mask_ratio": 0.75
}
```

Response:
```json
{
  "reconstruction": "<base64_encoded_reconstruction>",
  "mask_ratio": 0.75,
  "processing_time_ms": 42
}
```

### Batch Reconstruction
```bash
POST /reconstruct_batch
```

Request:
```json
{
  "images": ["<base64_image1>", "<base64_image2>", ...],
  "mask_ratio": 0.75
}
```

Response:
```json
{
  "reconstructions": ["<base64_recon1>", "<base64_recon2>", ...],
  "batch_size": 2,
  "mask_ratio": 0.75,
  "processing_time_ms": 85
}
```

## Testing the Server

Use the provided Python client:

```bash
# Test single image
python examples/test_client.py --image path/to/image.jpg --mask-ratio 0.75

# Test batch processing
python examples/test_client.py --batch img1.jpg img2.jpg img3.jpg

# Test remote server
python examples/test_client.py --host 192.168.1.100 --port 8080 --image test.jpg
```

## Performance Tips

1. **GPU Acceleration**: The server automatically uses CUDA if available
2. **Batch Processing**: Use the batch endpoint for multiple images to improve throughput
3. **Image Format**: PNG and JPEG images are supported
4. **Concurrency**: The server handles multiple concurrent requests

## Docker Deployment

Create a Dockerfile:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download LibTorch
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip \
    && rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

# Copy source code
COPY . /app
WORKDIR /app

# Build
RUN mkdir build && cd build \
    && cmake -DCMAKE_PREFIX_PATH=/libtorch .. \
    && cmake --build . -j$(nproc)

# Expose port
EXPOSE 8080

# Run server
CMD ["./build/mae_server", "--model", "checkpoints/mae_model.pt", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:
```bash
docker build -t mae-server .
docker run -p 8080:8080 --gpus all mae-server
```

## Integration Examples

### Python
```python
import requests
import base64

# Encode image
with open('image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post('http://localhost:8080/reconstruct', 
                        json={'image': image_base64, 'mask_ratio': 0.75})

# Decode result
result = response.json()
reconstruction_data = base64.b64decode(result['reconstruction'])
with open('reconstruction.png', 'wb') as f:
    f.write(reconstruction_data)
```

### JavaScript
```javascript
// Read and encode image
const imageBuffer = fs.readFileSync('image.jpg');
const imageBase64 = imageBuffer.toString('base64');

// Send request
const response = await fetch('http://localhost:8080/reconstruct', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        image: imageBase64,
        mask_ratio: 0.75
    })
});

// Decode result
const result = await response.json();
const reconstruction = Buffer.from(result.reconstruction, 'base64');
fs.writeFileSync('reconstruction.png', reconstruction);
```

### cURL
```bash
# Encode image
IMAGE_BASE64=$(base64 -w 0 image.jpg)

# Send request
curl -X POST http://localhost:8080/reconstruct \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"$IMAGE_BASE64\", \"mask_ratio\": 0.75}" \
    | jq -r .reconstruction \
    | base64 -d > reconstruction.png
```