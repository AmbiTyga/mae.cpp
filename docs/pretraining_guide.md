# MAE Pretraining Guide

This guide explains how to pretrain Masked Autoencoder (MAE) models using the C++ implementation.

## Overview

The pretraining setup follows the original MAE paper specifications:
- **Batch size**: 4096 (effective batch size with gradient accumulation)
- **Optimizer**: AdamW with β1=0.9, β2=0.95
- **Learning rate**: Base LR scaled with batch size (e.g., 2.4e-3 for ViT-Base)
- **Schedule**: Cosine decay with linear warmup
- **Weight decay**: 0.05
- **Data augmentation**: RandomResizedCrop (scale 0.2-1.0) + RandomHorizontalFlip
- **Normalization**: ImageNet statistics

## Configuration Files

Training is configured using JSON files in the `configs/` directory:

- `mae_pretrain_vit_base.json`: ViT-Base/16 configuration (1600 epochs)
- `mae_pretrain_vit_large.json`: ViT-Large/16 configuration (1600 epochs)
- `mae_pretrain_vit_base_test.json`: Test configuration with smaller batch size

### Configuration Structure

```json
{
    "model": {
        "type": "mae_vit_base_patch16",
        "norm_pix_loss": true,
        "mask_ratio": 0.75
    },
    "optimization": {
        "batch_size": 4096,
        "base_lr": 2.4e-3,
        "min_lr": 0.0,
        "weight_decay": 0.05,
        "optimizer": "adamw",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "gradient_clip": 1.0
    },
    "schedule": {
        "epochs": 1600,
        "warmup_epochs": 40,
        "start_epoch": 0
    },
    "data": {
        "data_path": "./data/imagenet/train",
        "input_size": 224,
        "num_workers": 10,
        "pin_memory": true,
        "augmentation": {...}
    },
    "checkpointing": {
        "checkpoint_dir": "./checkpoints",
        "save_freq_epochs": 50,
        "save_freq_steps": 10000,
        "keep_last_n": 5
    },
    "logging": {
        "print_freq": 100,
        "console_clear_freq": 100
    }
}
```

## Building

```bash
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . -j$(nproc)
```

## Pretraining

### Basic Usage

```bash
./pretrain_mae configs/mae_pretrain_vit_base.json
```

### With Custom Data Path

Edit the config file to update `data.data_path` or create a new config:

```json
{
    ...
    "data": {
        "data_path": "/path/to/imagenet/train",
        ...
    }
}
```

### Testing with Smaller Batch Size

For testing or running on limited GPU memory:

```bash
./pretrain_mae configs/mae_pretrain_vit_base_test.json
```

## Monitoring Training

### Logs

Training logs are saved to timestamped files:
- Console output: Clears every 100 lines to prevent memory issues
- Log files: `mae_pretrain_YYYYMMDD_HHMMSS.log` contains complete logs

### Checkpoints

Checkpoints are saved according to the configuration:
- **Epoch checkpoints**: `checkpoints/epoch-N/`
- **Step checkpoints**: `checkpoints/step-N/`
- **Latest checkpoint**: `checkpoints/latest.pt`
- **Final model**: `checkpoints/final.pt`

Each checkpoint directory contains:
- `model.pt`: Model weights and training state
- `config.json`: Training configuration
- `mae_pretrain_*.log`: Training log up to that point

### Metrics

The training script reports:
- Loss (smoothed and instantaneous)
- Learning rate
- Training speed (samples/second)
- Time per iteration
- ETA for epoch completion

## Resuming Training

### Automatic Resume

Set `"auto_resume": true` in the config to automatically resume from `checkpoints/latest.pt` if it exists.

### Manual Resume

```json
{
    "misc": {
        "resume": "checkpoints/epoch-200/model.pt",
        ...
    }
}
```

## Multi-GPU Training

Currently, the C++ implementation supports single-GPU training. For multi-GPU training, consider:
1. Using gradient accumulation to simulate larger batch sizes
2. Running multiple experiments with different seeds
3. Using the Python implementation for distributed training

## Learning Rate Scaling

The base learning rate should be scaled with the effective batch size:
```
lr = base_lr * batch_size / 256
```

For example:
- Batch size 256: lr = 1.5e-4
- Batch size 1024: lr = 6e-4
- Batch size 4096: lr = 2.4e-3

## Tips for Efficient Training

1. **Data Loading**: 
   - Use multiple workers (`num_workers`)
   - Enable pinned memory for GPU transfer
   - Ensure fast storage (SSD/NVMe) for dataset

2. **Memory Management**:
   - Console output clears automatically
   - Logs are flushed immediately
   - Checkpoints include only necessary data

3. **Batch Size**:
   - Use the largest batch size that fits in GPU memory
   - Consider gradient accumulation for larger effective batch sizes

4. **Monitoring**:
   ```bash
   # Watch GPU usage
   watch -n 1 nvidia-smi
   
   # Follow log file
   tail -f mae_pretrain_*.log
   
   # Check checkpoint sizes
   du -sh checkpoints/*/
   ```

## Converting to TorchScript

After pretraining, convert the model for inference:

```python
import torch
from mae_model import mae_vit_base_patch16_dec512d8b

# Load checkpoint
checkpoint = torch.load('checkpoints/final.pt')
model = mae_vit_base_patch16_dec512d8b()
model.load_state_dict(checkpoint)
model.eval()

# Trace for inference
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, (example_input, 0.0))  # 0.0 mask ratio for inference
traced_model.save('mae_inference.pt')
```

## Troubleshooting

### Out of Memory
- Reduce batch size in config
- Reduce number of workers
- Use gradient checkpointing (not implemented yet)

### Slow Data Loading
- Increase number of workers
- Check disk I/O performance
- Ensure dataset is on fast storage

### Training Divergence
- Check learning rate scaling
- Verify data augmentation
- Ensure proper weight initialization