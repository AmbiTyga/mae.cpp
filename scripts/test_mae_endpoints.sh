#!/bin/bash

# Test MAE Server Endpoints
SERVER="http://localhost:8080"

echo "Testing MAE Server Endpoints..."
echo "=============================="

# 1. Health check
echo -e "\n1. Health Check:"
curl -s $SERVER/health | jq .

# 2. Server info
echo -e "\n2. Server Info:"
curl -s $SERVER/info | jq .

# 3. Create a test image if it doesn't exist
if [ ! -f "test_image.jpg" ]; then
    echo -e "\n3. Creating test image..."
    python3 -c "
import numpy as np
import cv2
# Create a simple test image with colored squares
img = np.zeros((224, 224, 3), dtype=np.uint8)
colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
for i in range(2):
    for j in range(2):
        cv2.rectangle(img, (j*112, i*112), ((j+1)*112, (i+1)*112), colors[i*2+j], -1)
cv2.imwrite('test_image.jpg', img)
print('Created test_image.jpg')
"
fi

# 4. Test visualize patches
echo -e "\n4. Testing Visualize Patches:"
python3 -c "
import requests
import base64
import json

with open('test_image.jpg', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post('$SERVER/visualize_patches', json={
    'image': img_base64,
    'show_numbers': True
})

result = response.json()
if 'error' not in result:
    print(f\"Success! Total patches: {result.get('total_patches', 'N/A')}\")
else:
    print(f\"Error: {result['error']}\")
"

# 5. Test mask image
echo -e "\n5. Testing Mask Image (75% masking):"
python3 -c "
import requests
import base64
import json

with open('test_image.jpg', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post('$SERVER/mask_image', json={
    'image': img_base64,
    'mask_ratio': 0.75
})

result = response.json()
if 'error' not in result:
    print(f\"Success! Masked patches: {result.get('num_masked_patches', 'N/A')}\")
    print(f\"Visible patches: {result.get('num_visible_patches', 'N/A')}\")
    
    # Save masked image
    import cv2
    import numpy as np
    img_data = base64.b64decode(result['masked_image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('test_masked.png', img)
    print('Saved masked image to test_masked.png')
else:
    print(f\"Error: {result['error']}\")
"

# 6. Test reconstruction
echo -e "\n6. Testing Reconstruction:"
python3 -c "
import requests
import base64
import json

with open('test_image.jpg', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post('$SERVER/reconstruct', json={
    'image': img_base64,
    'mask_ratio': 0.75
})

result = response.json()
if 'error' not in result:
    print(f\"Success! Processing time: {result.get('processing_time_ms', 'N/A')}ms\")
    
    # Save reconstructed image
    import cv2
    import numpy as np
    img_data = base64.b64decode(result['reconstruction'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('test_reconstructed.png', img)
    print('Saved reconstructed image to test_reconstructed.png')
else:
    print(f\"Error: {result['error']}\")
"

echo -e "\n=============================="
echo "Testing complete!"
echo "Check the generated images:"
echo "  - test_image.jpg (original)"
echo "  - test_masked.png (masked version)"
echo "  - test_reconstructed.png (reconstruction)"