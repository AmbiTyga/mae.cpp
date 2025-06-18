# wget Workarounds for MAE Server

If you're getting 500 errors with wget, here are several workarounds:

## Option 1: Use curl Instead (Recommended)

```bash
curl -X POST http://localhost:8080/reconstruct/binary \
  -H "Content-Type: image/jpeg" \
  -H "X-Mask-Ratio: 0.75" \
  --data-binary "@image.jpg" \
  --output result.png
```

## Option 2: Use Python Script

Create `mae_wget.py`:
```python
#!/usr/bin/env python3
import sys
import requests

if len(sys.argv) < 2:
    print("Usage: python mae_wget.py <image_path> [mask_ratio]")
    sys.exit(1)

image_path = sys.argv[1]
mask_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75

with open(image_path, 'rb') as f:
    response = requests.post(
        'http://localhost:8080/reconstruct/binary',
        data=f.read(),
        headers={
            'Content-Type': 'image/jpeg',
            'X-Mask-Ratio': str(mask_ratio)
        }
    )

if response.status_code == 200:
    output_name = f"reconstructed_{image_path}"
    with open(output_name, 'wb') as f:
        f.write(response.content)
    print(f"Saved to {output_name}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

Usage:
```bash
python mae_wget.py image.jpg 0.75
```

## Option 3: Use HTTPie

Install HTTPie:
```bash
pip install httpie
```

Use it:
```bash
http POST localhost:8080/reconstruct/binary \
  Content-Type:image/jpeg \
  X-Mask-Ratio:0.75 \
  < image.jpg > result.png
```

## Option 4: wget with Proxy Workaround

Sometimes wget works better through a proxy:
```bash
# Use socat as a proxy
socat TCP-LISTEN:8081,fork TCP:localhost:8080 &

# Then use wget
wget --post-file=image.jpg \
     --header="Content-Type: image/jpeg" \
     --header="X-Mask-Ratio: 0.75" \
     -O result.png \
     http://localhost:8081/reconstruct/binary
```

## Option 5: wget Debug Mode

To understand what's failing:
```bash
# Maximum verbosity
wget --debug \
     --post-file=image.jpg \
     --header="Content-Type: image/jpeg" \
     --header="X-Mask-Ratio: 0.75" \
     -O result.png \
     http://localhost:8080/reconstruct/binary 2>&1 | tee wget_debug.log
```

## Option 6: Use multipart Instead

The multipart endpoint might work better:
```bash
# Create a simple form submission
cat > form.txt << EOF
--boundary
Content-Disposition: form-data; name="image"; filename="image.jpg"
Content-Type: image/jpeg

$(cat image.jpg)
--boundary
Content-Disposition: form-data; name="mask_ratio"

0.75
--boundary--
EOF

wget --post-file=form.txt \
     --header="Content-Type: multipart/form-data; boundary=boundary" \
     -O result.png \
     http://localhost:8080/reconstruct/multipart
```

## Debugging the Issue

1. **Check server logs**:
```bash
tail -f mae_server_*.log | grep -E "binary|ERROR"
```

2. **Check if wget sends empty body**:
```bash
# Start a netcat listener
nc -l 8888 > request.txt &

# Send wget request to netcat
wget --post-file=image.jpg http://localhost:8888/test

# Check what was sent
cat request.txt
```

3. **Common wget issues**:
- wget might not send Content-Length header
- wget might chunk the request differently
- Some wget versions have bugs with --post-file

## Recommended Solution

For production use, we recommend using curl or creating a simple client script in Python. wget's HTTP POST implementation has known quirks that can cause issues with binary data.