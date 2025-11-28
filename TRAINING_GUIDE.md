# TryOnDiffusion Training Guide

This guide will walk you through the complete process of training TryOnDiffusion models from scratch.

## Prerequisites

### System Requirements

- Windows/Linux with Python 3.8+
- NVIDIA GPU with at least 16GB VRAM (24GB+ recommended)
- 64GB+ RAM
- 500GB+ storage space

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 1: Dataset Preparation

### 1.1 Create Dataset Directory Structure

```
datasets/
├── images/
│   ├── person/          # Person images
│   ├── garment/         # Garment images
│   └── ca_images/       # Clothing-agnostic images
├── poses/
│   ├── person_poses/    # Person pose keypoints (.json or .txt)
│   └── garment_poses/   # Garment pose keypoints
└── metadata/
    ├── train.csv        # Training metadata
    └── val.csv          # Validation metadata
```

### 1.2 Dataset Format Requirements

**Images:**

- Format: PNG, JPG
- Size: 512x512 or 1024x1024 (will be resized during training)
- Person images: Full body images of people
- Garment images: Clean garment images on white/transparent background
- CA (Clothing-Agnostic) images: Person with target clothing area masked/removed

**Poses:**

- Format: JSON files with 18 keypoints (COCO format)
- Each keypoint: [x, y] coordinates
- Keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

**Metadata CSV format:**

```csv
person_image,ca_image,garment_image,person_pose,garment_pose
person/001.jpg,ca_images/001.jpg,garment/001.jpg,person_poses/001.json,garment_poses/001.json
```

## Step 2: Create Custom Dataset Class

Create `datasets/tryon_dataset.py`:

```python
import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TryOnDataset(Dataset):
    def __init__(self, csv_file, dataset_root, image_size=(128, 128), transform=None):
        self.df = pd.read_csv(csv_file)
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.transform = transform or self.get_default_transform()

    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def load_pose(self, pose_path):
        with open(os.path.join(self.dataset_root, pose_path), 'r') as f:
            pose_data = json.load(f)
        # Extract keypoints and convert to tensor
        keypoints = pose_data['keypoints']  # Assuming COCO format
        pose_tensor = torch.tensor(keypoints).reshape(-1, 2).float()
        return pose_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load images
        person_img = Image.open(os.path.join(self.dataset_root, row['person_image'])).convert('RGB')
        ca_img = Image.open(os.path.join(self.dataset_root, row['ca_image'])).convert('RGB')
        garment_img = Image.open(os.path.join(self.dataset_root, row['garment_image'])).convert('RGB')

        # Load poses
        person_pose = self.load_pose(row['person_pose'])
        garment_pose = self.load_pose(row['garment_pose'])

        # Apply transforms
        person_img = self.transform(person_img)
        ca_img = self.transform(ca_img)
        garment_img = self.transform(garment_img)

        return {
            'person_images': person_img,
            'ca_images': ca_img,
            'garment_images': garment_img,
            'person_poses': person_pose,
            'garment_poses': garment_pose
        }

def collate_fn(batch):
    return {
        'person_images': torch.stack([item['person_images'] for item in batch]),
        'ca_images': torch.stack([item['ca_images'] for item in batch]),
        'garment_images': torch.stack([item['garment_images'] for item in batch]),
        'person_poses': torch.stack([item['person_poses'] for item in batch]),
        'garment_poses': torch.stack([item['garment_poses'] for item in batch]),
    }
```

## Step 3: Training Configuration

Create `configs/train_config.py`:

```python
# Training Configuration
TRAIN_CONFIG = {
    # Dataset
    'dataset_root': 'datasets/',
    'train_csv': 'datasets/metadata/train.csv',
    'val_csv': 'datasets/metadata/val.csv',

    # Model
    'base_image_size': (128, 128),
    'sr_image_size': (256, 256),
    'timesteps': (1000, 250),  # Base UNet, SR UNet timesteps

    # Training
    'batch_size': 4,  # Adjust based on GPU memory
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'gradient_accumulation_steps': 4,
    'max_grad_norm': 1.0,

    # Training stages
    'train_base_unet_epochs': 50,    # Train base UNet first
    'train_sr_unet_epochs': 50,      # Then train SR UNet

    # Checkpointing
    'checkpoint_every': 1000,
    'checkpoint_path': './checkpoints',
    'resume_from_checkpoint': None,

    # Validation
    'validate_every': 2000,
    'num_validation_samples': 4,

    # Hardware
    'use_fp16': True,
    'use_ema': True,
    'num_workers': 4,
}
```

## Step 4: Training Script

Create `train.py`:

```python
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tryondiffusion import TryOnImagen, TryOnImagenTrainer, get_unet_by_name
from datasets.tryon_dataset import TryOnDataset, collate_fn
from configs.train_config import TRAIN_CONFIG

def setup_datasets():
    # Create datasets
    train_dataset = TryOnDataset(
        csv_file=TRAIN_CONFIG['train_csv'],
        dataset_root=TRAIN_CONFIG['dataset_root'],
        image_size=TRAIN_CONFIG['base_image_size']
    )

    val_dataset = TryOnDataset(
        csv_file=TRAIN_CONFIG['val_csv'],
        dataset_root=TRAIN_CONFIG['dataset_root'],
        image_size=TRAIN_CONFIG['base_image_size']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader

def setup_model():
    # Create UNets
    base_unet = get_unet_by_name("base")
    sr_unet = get_unet_by_name("sr")

    # Create Imagen model
    imagen = TryOnImagen(
        unets=(base_unet, sr_unet),
        image_sizes=(TRAIN_CONFIG['base_image_size'], TRAIN_CONFIG['sr_image_size']),
        timesteps=TRAIN_CONFIG['timesteps'],
    )

    return imagen

def train_stage(trainer, unet_number, num_epochs, stage_name):
    print(f"\\n=== Training {stage_name} ===")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        total_loss = 0
        num_batches = 0

        for batch_idx in range(len(trainer.train_dataloader)):
            loss = trainer.train_step(unet_number=unet_number)
            total_loss += loss
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx+1}, Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}")

            # Validation
            if (batch_idx + 1) % TRAIN_CONFIG['validate_every'] == 0:
                val_loss = trainer.valid_step(unet_number=unet_number)
                print(f"  Validation Loss: {val_loss:.4f}")

        # Save checkpoint after each epoch
        trainer.save(f"{TRAIN_CONFIG['checkpoint_path']}/{stage_name}_epoch_{epoch+1}")

def main():
    print("Setting up datasets...")
    train_loader, val_loader = setup_datasets()

    print("Setting up model...")
    imagen = setup_model()

    print("Setting up trainer...")
    trainer = TryOnImagenTrainer(
        imagen=imagen,
        lr=TRAIN_CONFIG['learning_rate'],
        max_grad_norm=TRAIN_CONFIG['max_grad_norm'],
        checkpoint_every=TRAIN_CONFIG['checkpoint_every'],
        checkpoint_path=TRAIN_CONFIG['checkpoint_path'],
        use_ema=TRAIN_CONFIG['use_ema'],
        fp16=TRAIN_CONFIG['use_fp16'],
        accelerate_gradient_accumulation_steps=TRAIN_CONFIG['gradient_accumulation_steps'],
    )

    trainer.add_train_dataloader(train_loader)
    trainer.add_valid_dataloader(val_loader)

    # Create checkpoint directory
    os.makedirs(TRAIN_CONFIG['checkpoint_path'], exist_ok=True)

    # Resume from checkpoint if specified
    if TRAIN_CONFIG['resume_from_checkpoint']:
        trainer.load(TRAIN_CONFIG['resume_from_checkpoint'])
        print(f"Resumed from checkpoint: {TRAIN_CONFIG['resume_from_checkpoint']}")

    # Stage 1: Train Base UNet
    train_stage(trainer, unet_number=1,
               num_epochs=TRAIN_CONFIG['train_base_unet_epochs'],
               stage_name="base_unet")

    # Stage 2: Train SR UNet
    train_stage(trainer, unet_number=2,
               num_epochs=TRAIN_CONFIG['train_sr_unet_epochs'],
               stage_name="sr_unet")

    print("\\nTraining completed!")

if __name__ == "__main__":
    main()
```

## Step 5: Training Commands

### 5.1 Prepare Your Dataset

```bash
# Create required directories
mkdir -p datasets/images/person
mkdir -p datasets/images/garment
mkdir -p datasets/images/ca_images
mkdir -p datasets/poses/person_poses
mkdir -p datasets/poses/garment_poses
mkdir -p datasets/metadata

# Place your images and poses in respective directories
# Create train.csv and val.csv with proper metadata
```

### 5.2 Start Training

```bash
# From the project root directory
python train.py
```

### 5.3 Monitor Training

```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f checkpoints/training.log
```

## Step 6: Inference/Sampling

Create `sample.py`:

```python
import torch
from PIL import Image
from tryondiffusion import TryOnImagen, get_unet_by_name

def load_trained_model(checkpoint_path):
    # Load UNets
    base_unet = get_unet_by_name("base")
    sr_unet = get_unet_by_name("sr")

    # Create Imagen
    imagen = TryOnImagen(
        unets=(base_unet, sr_unet),
        image_sizes=((128, 128), (256, 256)),
        timesteps=(1000, 250),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    imagen.load_state_dict(checkpoint['imagen'])

    return imagen

def sample_images(imagen, ca_images, garment_images, person_poses, garment_poses):
    with torch.no_grad():
        images = imagen.sample(
            ca_images=ca_images,
            garment_images=garment_images,
            person_poses=person_poses,
            garment_poses=garment_poses,
            batch_size=1,
            cond_scale=3.0,
            start_at_unet_number=1,
            return_all_unet_outputs=True,
            return_pil_images=True,
            use_tqdm=True,
        )
    return images

# Usage
imagen = load_trained_model("checkpoints/sr_unet_epoch_50.pth")
# Load your test data and run sampling
```

## Tips for Successful Training

1. **Start Small**: Begin with a smaller dataset (1K-10K images) to test the pipeline
2. **Monitor Memory**: Adjust batch size based on GPU memory
3. **Use Mixed Precision**: Enable FP16 to save memory
4. **Gradient Accumulation**: Use when you need larger effective batch sizes
5. **Regular Checkpointing**: Save frequently to avoid losing progress
6. **Validation Monitoring**: Watch for overfitting

## Troubleshooting

- **Out of Memory**: Reduce batch size, enable gradient checkpointing
- **Slow Training**: Increase num_workers, use faster storage (SSD)
- **Poor Results**: Check data quality, increase training time, tune hyperparameters
