# Dataset Setup Guide

This guide explains how to prepare and organize datasets for training the MAE model.

## Dataset Structure

The implementation expects datasets to be organized in the ImageFolder format:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    └── class2/
        ├── image1.jpg
        └── ...
```

## 1. ImageNet Dataset

### Download ImageNet

1. Register at http://image-net.org
2. Download ILSVRC2012 dataset:
   - Training images (138GB): `ILSVRC2012_img_train.tar`
   - Validation images (6.3GB): `ILSVRC2012_img_val.tar`

### Prepare ImageNet

```bash
# Create directories
mkdir -p data/imagenet/{train,val}

# Extract training data
cd data/imagenet/train
tar -xf /path/to/ILSVRC2012_img_train.tar

# Extract training tar files (each class is a tar file)
for f in *.tar; do
  d="${f%.tar}"
  mkdir -p "$d"
  tar -xf "$f" -C "$d"
  rm "$f"
done

# Extract validation data
cd ../val
tar -xf /path/to/ILSVRC2012_img_val.tar

# Organize validation data into folders
# Download validation ground truth
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
```

### Alternative: Use ImageNet subset

For testing purposes, you can use ImageNet-100 (100 classes subset):

```bash
# Create subset with 100 random classes
import os
import random
import shutil

# Select 100 random classes
all_classes = os.listdir('data/imagenet/train')
selected_classes = random.sample(all_classes, 100)

# Copy selected classes
for cls in selected_classes:
    shutil.copytree(f'data/imagenet/train/{cls}', 
                    f'data/imagenet100/train/{cls}')
    if os.path.exists(f'data/imagenet/val/{cls}'):
        shutil.copytree(f'data/imagenet/val/{cls}', 
                        f'data/imagenet100/val/{cls}')
```

## 2. Custom Dataset Preparation

### Organize your images:

```python
import os
import shutil
from pathlib import Path

def organize_dataset(source_dir, target_dir, train_split=0.8):
    """
    Organize images into train/val splits
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create directories
    (target_path / 'train').mkdir(parents=True, exist_ok=True)
    (target_path / 'val').mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_dir in source_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            images = list(class_dir.glob('*.jpg')) + \
                    list(class_dir.glob('*.png')) + \
                    list(class_dir.glob('*.jpeg'))
            
            # Shuffle and split
            random.shuffle(images)
            split_idx = int(len(images) * train_split)
            
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Copy to train
            train_class_dir = target_path / 'train' / class_name
            train_class_dir.mkdir(exist_ok=True)
            for img in train_images:
                shutil.copy2(img, train_class_dir)
            
            # Copy to val
            val_class_dir = target_path / 'val' / class_name
            val_class_dir.mkdir(exist_ok=True)
            for img in val_images:
                shutil.copy2(img, val_class_dir)
            
            print(f"Class {class_name}: {len(train_images)} train, "
                  f"{len(val_images)} val")
```

## 3. Data Augmentation

The C++ implementation includes basic augmentation. For advanced augmentation, preprocess your dataset:

```python
from PIL import Image
import torchvision.transforms as transforms

# Define augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Apply augmentation and save
def augment_dataset(input_dir, output_dir, num_augmentations=5):
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert('RGB')
            
            # Save original
            img.save(os.path.join(output_class_path, img_name))
            
            # Save augmented versions
            for i in range(num_augmentations):
                aug_img = train_transform(img)
                aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
                # Convert tensor back to PIL image
                aug_img_pil = transforms.ToPILImage()(aug_img)
                aug_img_pil.save(os.path.join(output_class_path, aug_name))
```

## 4. Running Training with Dataset

Once your dataset is prepared:

```bash
# Build the project (see pytorch_cpp_setup.md)
cd build

# Run training
./train_mae /path/to/data/train [batch_size] [epochs]

# Example with ImageNet
./train_mae ../data/imagenet/train 64 400

# Example with custom dataset
./train_mae ../data/my_dataset/train 32 200
```

## 5. Dataset Statistics

For optimal training, calculate your dataset statistics:

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def calculate_dataset_stats(data_dir):
    """Calculate mean and std of dataset"""
    dataset = datasets.ImageFolder(
        data_dir,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    )
    
    dataloader = DataLoader(dataset, batch_size=100, 
                          shuffle=False, num_workers=4)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    return mean, std

# Calculate for your dataset
mean, std = calculate_dataset_stats('data/my_dataset/train')
```

## 6. Memory Considerations

For RTX 4090 (24GB VRAM):

- **ImageNet (224x224)**: Batch size 64-128
- **Higher resolution (384x384)**: Batch size 16-32
- **Large models (ViT-Large/Huge)**: Reduce batch size accordingly

### Estimate memory usage:
```
Memory ≈ (batch_size × 3 × height × width × 4) + model_parameters × 4
```

## 7. Multi-GPU Dataset Loading

For multi-GPU training (future enhancement), organize data sharding:

```bash
# Split dataset for distributed training
data/
├── node0/
│   └── imagenet/
├── node1/
│   └── imagenet/
└── ...
```

## 8. Data Validation

Validate your dataset before training:

```bash
# Create validation script
cat > validate_dataset.py << 'EOF'
import os
from PIL import Image
import sys

def validate_dataset(data_dir):
    """Check dataset for common issues"""
    issues = []
    total_images = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
                filepath = os.path.join(root, file)
                
                try:
                    img = Image.open(filepath)
                    img.verify()
                    
                    # Check image properties
                    if img.mode != 'RGB':
                        issues.append(f"Non-RGB image: {filepath}")
                    
                    width, height = img.size
                    if width < 224 or height < 224:
                        issues.append(f"Image too small: {filepath} ({width}x{height})")
                        
                except Exception as e:
                    issues.append(f"Corrupt image: {filepath} - {str(e)}")
    
    print(f"Total images: {total_images}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\nFirst 10 issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    
    return len(issues) == 0

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    validate_dataset(data_dir)
EOF

python validate_dataset.py data/train
```

## Tips for Best Results

1. **Use high-quality images**: Remove blurry, corrupt, or very small images
2. **Balance classes**: Ensure reasonable distribution across classes
3. **Augmentation**: Use appropriate data augmentation for your domain
4. **Preprocessing**: Consistent preprocessing improves convergence
5. **Caching**: For small datasets, consider caching in RAM:
   ```bash
   # Increase system file cache
   sudo sysctl -w vm.vfs_cache_pressure=10
   ```