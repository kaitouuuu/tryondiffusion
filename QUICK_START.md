# Quick Start Training Commands for TryOnDiffusion

## Test with Synthetic Data (No Real Dataset Required)

```powershell
# Test the pipeline with synthetic data first
python train.py --synthetic --config low_memory --unet-stage 1
```

## Training with Real Dataset

### Step 1: Prepare your dataset

1. Place your images in the correct folder structure:

   ```
   datasets/
   ├── images/
   │   ├── person/          # Your person images
   │   ├── garment/         # Your garment images
   │   └── ca_images/       # Your clothing-agnostic images
   ├── poses/
   │   ├── person_poses/    # Person pose JSON files
   │   └── garment_poses/   # Garment pose JSON files
   └── metadata/
       ├── train.csv        # Training file paths
       └── val.csv          # Validation file paths
   ```

2. Update the CSV files with your actual file paths

### Step 2: Choose your hardware configuration

- `--config low_memory`: For GPUs with 8GB VRAM
- `--config medium_memory`: For GPUs with 16GB VRAM
- `--config high_memory`: For GPUs with 24GB+ VRAM

### Step 3: Train Base UNet (Stage 1)

```powershell
python train.py --config medium_memory --unet-stage 1
```

### Step 4: Train Super-Resolution UNet (Stage 2)

```powershell
# After base UNet training completes
python train.py --config medium_memory --unet-stage 2 --resume checkpoints/base_unet_best.pth
```

### Step 5: Resume Training (if interrupted)

```powershell
python train.py --config medium_memory --unet-stage 1 --resume checkpoints/base_unet_epoch_10.pth
```

## Sampling/Inference

```powershell
python sample.py --checkpoint checkpoints/sr_unet_best.pth --ca-image test_ca.jpg --garment-image test_garment.jpg --person-pose test_person_pose.json --garment-pose test_garment_pose.json --output result.png
```

## Monitor Training

- Check `checkpoints/` folder for saved models
- View sample images in `checkpoints/samples/`
- Monitor GPU usage: `nvidia-smi`
- Watch training logs in terminal

## Training Tips

1. Start with synthetic data to test the pipeline
2. Use the appropriate config for your GPU memory
3. Base UNet should be trained first, then SR UNet
4. Training can take days/weeks depending on dataset size
5. Monitor validation loss to avoid overfitting
6. Save checkpoints frequently
