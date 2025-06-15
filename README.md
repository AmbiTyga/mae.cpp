# Masked Autoencoder (MAE) - C++ Implementation

A PyTorch C++ implementation of Masked Autoencoder (MAE) for self-supervised visual representation learning.

## Overview

This project implements the Masked Autoencoder (MAE) architecture from the paper "Masked Autoencoders Are Scalable Vision Learners" using PyTorch C++ API (LibTorch). The implementation is optimized for training on modern GPUs like RTX 4090.

## Features

- Full MAE architecture implementation in C++
- Support for ViT-Base, ViT-Large, and ViT-Huge variants
- Efficient data loading with OpenCV
- Mixed precision training support
- Checkpoint saving/loading
- Cosine learning rate schedule with warmup

## Project Structure

```
MAE/
├── include/
│   ├── mae_model.hpp      # MAE model definition
│   └── data_loader.hpp    # Dataset and data loading utilities
├── src/
│   ├── mae_model.cpp      # MAE model implementation
│   ├── data_loader.cpp    # Dataset implementation
│   └── train_mae.cpp      # Training script
├── docs/
│   ├── pytorch_cpp_setup.md  # PyTorch C++ setup guide
│   └── dataset_setup.md      # Dataset preparation guide
├── CMakeLists.txt         # CMake build configuration
└── README.md             # This file
```

## Requirements

- CMake 3.14+
- C++17 compatible compiler
- PyTorch C++ (LibTorch) 2.0+
- OpenCV 4.x
- CUDA 11.x or 12.x (for GPU support)

## Quick Start

### 1. Setup PyTorch C++

Follow the detailed guide in `docs/pytorch_cpp_setup.md` to install LibTorch.

```bash
# Download LibTorch (example for CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
```

### 2. Build the Project

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release -j$(nproc)
```

### 3. Prepare Dataset

Follow `docs/dataset_setup.md` to prepare your dataset in ImageFolder format:

```
data/
└── train/
    ├── class1/
    │   ├── img1.jpg
    │   └── ...
    └── class2/
        ├── img1.jpg
        └── ...
```

### 4. Train the Model

```bash
./train_mae /path/to/data/train [batch_size] [epochs]

# Example
./train_mae ../data/imagenet/train 64 400
```

## Model Variants

The implementation includes three model variants:

| Model | Patches | Embed Dim | Heads | Layers | Params |
|-------|---------|-----------|-------|---------|---------|
| ViT-Base | 16x16 | 768 | 12 | 12 | 86M |
| ViT-Large | 16x16 | 1024 | 16 | 24 | 304M |
| ViT-Huge | 14x14 | 1280 | 16 | 32 | 632M |

## Training Configuration

Key training parameters (modify in `train_mae.cpp`):

```cpp
struct TrainingConfig {
    std::string model_type = "mae_vit_base_patch16";
    bool norm_pix_loss = true;
    int64_t batch_size = 64;
    int64_t epochs = 400;
    double learning_rate = 1.5e-4;
    double weight_decay = 0.05;
    double mask_ratio = 0.75;
    int64_t warmup_epochs = 40;
};
```

## Memory Requirements

Approximate VRAM usage for different configurations:

| Model | Batch Size | Resolution | VRAM Usage |
|-------|------------|------------|------------|
| ViT-Base | 64 | 224x224 | ~12GB |
| ViT-Base | 128 | 224x224 | ~20GB |
| ViT-Large | 32 | 224x224 | ~16GB |
| ViT-Large | 64 | 224x224 | ~24GB |

## Performance Tips

1. **Enable CUDNN Benchmark**: Automatically enabled in the code
2. **Use Mixed Precision**: Add autocast for faster training
3. **Optimize Batch Size**: Use the largest batch size that fits in memory
4. **Data Loading**: Increase number of workers for faster data loading

## Extending the Code

### Adding New Models

Create new model variants in `mae_model.cpp`:

```cpp
MaskedAutoencoderViT mae_vit_custom(bool norm_pix_loss) {
    return MaskedAutoencoderViT(
        224,    // img_size
        16,     // patch_size
        3,      // in_chans
        768,    // embed_dim
        12,     // depth
        12,     // num_heads
        512,    // decoder_embed_dim
        8,      // decoder_depth
        16,     // decoder_num_heads
        4.,     // mlp_ratio
        norm_pix_loss
    );
}
```

### Custom Data Augmentation

Extend the `ImageFolderDataset` class in `data_loader.cpp` to add custom augmentations.

## Citation

If you use this implementation, please cite the original MAE paper:

```bibtex
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={CVPR},
  year={2022}
}
```

## License

This implementation follows the license of the original MAE repository.