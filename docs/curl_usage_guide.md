# MAE Server cURL Usage Guide

This guide shows different ways to send images to the MAE server without dealing with long base64 strings.

## Method 1: Multipart Form Upload (Recommended for cURL)

This is the easiest method for cURL - directly upload the image file:

```bash
# Basic reconstruction with default 75% masking
curl -X POST http://localhost:8080/reconstruct/multipart \
  -F "image=@/path/to/your/image.jpg" \
  -F "mask_ratio=0.75" \
  -o reconstructed.png

# With custom mask ratio
curl -X POST http://localhost:8080/reconstruct/multipart \
  -F "image=@photo.jpg" \
  -F "mask_ratio=0.9" \
  -o reconstructed_90.png

# Save JSON response with base64 image
curl -X POST http://localhost:8080/reconstruct/multipart \
  -F "image=@photo.jpg" \
  -F "mask_ratio=0.75" \
  > response.json
```

## Method 2: Binary Upload/Download (Most Efficient)

Send raw image bytes and receive raw image bytes - no base64 encoding:

```bash
# Upload image as binary, get reconstructed image as binary
curl -X POST http://localhost:8080/reconstruct/binary \
  -H "Content-Type: image/jpeg" \
  -H "X-Mask-Ratio: 0.75" \
  --data-binary "@photo.jpg" \
  --output reconstructed.png

# With different mask ratios
curl -X POST http://localhost:8080/reconstruct/binary \
  -H "Content-Type: image/png" \
  -H "X-Mask-Ratio: 0.5" \
  --data-binary "@input.png" \
  --output output_50.png
```

## Method 3: Using wget Instead of cURL

```bash
# Using wget with binary data
wget --post-file=image.jpg \
     --header="Content-Type: image/jpeg" \
     --header="X-Mask-Ratio: 0.75" \
     -O reconstructed.png \
     http://localhost:8080/reconstruct/binary
```

## Method 4: Using HTTPie (More User-Friendly)

First install HTTPie:
```bash
pip install httpie
```

Then use it:
```bash
# Multipart upload
http -f POST localhost:8080/reconstruct/multipart \
  image@photo.jpg \
  mask_ratio=0.75 \
  > response.json

# Binary upload
http POST localhost:8080/reconstruct/binary \
  Content-Type:image/jpeg \
  X-Mask-Ratio:0.75 \
  < photo.jpg \
  > reconstructed.png
```

## Method 5: Using Python Requests (No Base64)

```python
import requests

# Method 1: Multipart
with open('photo.jpg', 'rb') as f:
    files = {'image': f}
    data = {'mask_ratio': '0.75'}
    response = requests.post('http://localhost:8080/reconstruct/multipart', 
                           files=files, data=data)
    
    # Extract base64 from JSON and decode
    import base64
    import json
    result = response.json()
    img_data = base64.b64decode(result['reconstruction'])
    with open('output.png', 'wb') as out:
        out.write(img_data)

# Method 2: Binary (simpler)
with open('photo.jpg', 'rb') as f:
    headers = {
        'Content-Type': 'image/jpeg',
        'X-Mask-Ratio': '0.75'
    }
    response = requests.post('http://localhost:8080/reconstruct/binary',
                           data=f.read(), headers=headers)
    
    # Response is already binary image
    with open('output.png', 'wb') as out:
        out.write(response.content)
```

## Method 6: Batch Processing Script

Create a script `process_images.sh`:

```bash
#!/bin/bash

# Process all images in a directory
for img in *.jpg *.png; do
    if [ -f "$img" ]; then
        echo "Processing $img..."
        curl -s -X POST http://localhost:8080/reconstruct/binary \
          -H "Content-Type: image/jpeg" \
          -H "X-Mask-Ratio: 0.75" \
          --data-binary "@$img" \
          --output "reconstructed_${img}"
    fi
done
```

## Method 7: Using Form HTML (For Web Interface)

Create a simple HTML file for browser-based uploads:

```html
<!DOCTYPE html>
<html>
<head>
    <title>MAE Upload</title>
</head>
<body>
    <h2>MAE Image Reconstruction</h2>
    <form action="http://localhost:8080/reconstruct/multipart" 
          method="post" enctype="multipart/form-data">
        <label>Select image:</label>
        <input type="file" name="image" accept="image/*" required><br><br>
        
        <label>Mask ratio:</label>
        <input type="number" name="mask_ratio" value="0.75" 
               min="0" max="1" step="0.05"><br><br>
        
        <input type="submit" value="Reconstruct">
    </form>
</body>
</html>
```

## Comparison of Methods

| Method | Pros | Cons |
|--------|------|------|
| Multipart | Easy with cURL, standard HTTP | Response still base64 |
| Binary | Most efficient, no encoding | Need to handle headers |
| Original JSON | Works everywhere | Long base64 strings |

## Advanced Examples

### Multiple Resolutions with Binary

```bash
# Test different sizes
for size in 224 448 672; do
    curl -X POST http://localhost:8080/reconstruct_multisize \
      -H "Content-Type: application/json" \
      -d "{\"image\":\"$(base64 -w 0 photo.jpg)\",\"size\":$size}" \
      | jq -r '.reconstruction' | base64 -d > "output_${size}.png"
done
```

### Piping with ImageMagick

```bash
# Resize, reconstruct, and convert in one pipeline
convert input.jpg -resize 512x512 jpg:- | \
  curl -s -X POST http://localhost:8080/reconstruct/binary \
    -H "Content-Type: image/jpeg" \
    -H "X-Mask-Ratio: 0.75" \
    --data-binary @- | \
  convert - -resize 1024x1024 output_large.png
```

### Error Handling

```bash
# With error checking
response=$(curl -s -w "\n%{http_code}" -X POST \
  http://localhost:8080/reconstruct/binary \
  -H "Content-Type: image/jpeg" \
  -H "X-Mask-Ratio: 0.75" \
  --data-binary "@photo.jpg")

http_code=$(echo "$response" | tail -n1)
content=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    echo "$content" > reconstructed.png
    echo "Success! Saved to reconstructed.png"
else
    echo "Error: HTTP $http_code"
    echo "$content"
fi
```

## Performance Tips

1. **Binary endpoints** are fastest - no base64 encoding/decoding
2. **Multipart** is convenient for cURL but adds overhead
3. **Keep connections alive** for multiple requests:
   ```bash
   # Use connection reuse
   curl -H "Connection: keep-alive" ...
   ```

4. **Compress large images** before sending:
   ```bash
   convert large.jpg -quality 85 -resize 2048x2048\> compressed.jpg
   ```

## Troubleshooting

### Image Too Large
```bash
# Check file size
ls -lh image.jpg

# Compress if needed
convert image.jpg -quality 80 smaller.jpg
```

### Server Timeout
```bash
# Increase timeout
curl --max-time 300 ...
```

### Debug Mode
```bash
# See what's being sent
curl -v -X POST ...
```