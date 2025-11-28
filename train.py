import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tryondiffusion import TryOnImagen, TryOnImagenTrainer, get_unet_by_name
from datasets.tryon_dataset import TryOnDataset, SyntheticTryOnDataset, collate_fn
from configs.train_config import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train TryOnDiffusion models')
    parser.add_argument('--config', type=str, default='medium_memory', 
                       choices=['low_memory', 'medium_memory', 'high_memory'],
                       help='Hardware configuration to use')
    parser.add_argument('--synthetic', action='store_true', 
                       help='Use synthetic dataset for testing')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--unet-stage', type=int, choices=[1, 2], default=1,
                       help='Which UNet to train (1=base, 2=SR)')
    return parser.parse_args()

def setup_datasets(config, use_synthetic=False):
    """Set up training and validation datasets"""
    if use_synthetic:
        print("Using synthetic dataset for testing...")
        train_dataset = SyntheticTryOnDataset(
            num_samples=1000,
            image_size=config['base_image_size']
        )
        val_dataset = SyntheticTryOnDataset(
            num_samples=100,
            image_size=config['base_image_size']
        )
    else:
        print("Loading real datasets...")
        train_dataset = TryOnDataset(
            csv_file=config['train_csv'],
            dataset_root=config['dataset_root'],
            image_size=config['base_image_size']
        )
        
        val_dataset = TryOnDataset(
            csv_file=config['val_csv'],
            dataset_root=config['dataset_root'],
            image_size=config['base_image_size']
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config.get('pin_memory', True)
    )
    
    return train_loader, val_loader

def setup_model(config):
    """Set up the TryOnImagen model with UNets"""
    print("Setting up model...")
    
    # Create UNets
    base_unet = get_unet_by_name("base")
    sr_unet = get_unet_by_name("sr")
    
    print(f"Base UNet parameters: {sum(p.numel() for p in base_unet.parameters()) / 1e6:.1f}M")
    print(f"SR UNet parameters: {sum(p.numel() for p in sr_unet.parameters()) / 1e6:.1f}M")
    
    # Create Imagen model
    imagen = TryOnImagen(
        unets=(base_unet, sr_unet),
        image_sizes=(config['base_image_size'], config['sr_image_size']),
        timesteps=config['timesteps'],
    )
    
    return imagen

def train_stage(trainer, unet_number, num_epochs, config, stage_name):
    """Train a specific UNet stage"""
    print(f"\\n=== Training {stage_name} (UNet {unet_number}) ===")
    
    best_val_loss = float('inf')
    
    # Calculate steps per epoch
    steps_per_epoch = len(trainer.train_dl)
    total_steps = steps_per_epoch * num_epochs
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    
    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        trainer.imagen.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        # Create a progress bar for this epoch
        epoch_iterator = tqdm(range(steps_per_epoch), desc=f"Training Epoch {epoch+1}")
        
        for step in epoch_iterator:
            try:
                loss = trainer.train_step(unet_number=unet_number)
                total_loss += loss
                num_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                epoch_iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}'
                })
                
                # Periodic logging
                if (step + 1) % config['log_every'] == 0:
                    elapsed = time.time() - start_time
                    print(f"  Step {step+1}/{steps_per_epoch}, "
                          f"Loss: {loss:.4f}, Avg: {avg_loss:.4f}, "
                          f"Time: {elapsed:.1f}s")
                
                # Validation
                if (step + 1) % config['validate_every'] == 0:
                    trainer.imagen.eval()
                    with torch.no_grad():
                        val_loss = trainer.valid_step(unet_number=unet_number)
                        print(f"  Validation Loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            trainer.save(f"{config['checkpoint_path']}/{stage_name}_best")
                            print(f"  New best validation loss: {val_loss:.4f}")
                    trainer.imagen.train()
                
                # Sample images
                if (step + 1) % config['sample_every'] == 0:
                    sample_images(trainer, unet_number, config, epoch, step)
                    
            except Exception as e:
                print(f"Error in training step: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # End of epoch
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
        
        # Save checkpoint after each epoch
        if (epoch + 1) % config['save_every_n_epochs'] == 0:
            checkpoint_path = f"{config['checkpoint_path']}/{stage_name}_epoch_{epoch+1}"
            trainer.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

def sample_images(trainer, unet_number, config, epoch, batch_idx):
    """Generate sample images for monitoring training progress"""
    try:
        trainer.imagen.eval()
        with torch.no_grad():
            validation_sample = next(trainer.valid_dl_iter)
            # Remove person_images for sampling (we only condition on CA images)
            person_images = validation_sample.pop("person_images", None)
            
            sample_kwargs = {
                **validation_sample,
                'batch_size': min(config['num_sample_images'], config['batch_size']),
                'cond_scale': config['cond_scale'],
                'start_at_unet_number': unet_number,
                'return_all_unet_outputs': False,
                'return_pil_images': True,
                'use_tqdm': False,
                'use_one_unet_in_gpu': True,
            }
            
            images = trainer.sample(**sample_kwargs)
            
            # Save sample images
            sample_dir = f"{config['checkpoint_path']}/samples"
            os.makedirs(sample_dir, exist_ok=True)
            
            for i, img in enumerate(images):
                img.save(f"{sample_dir}/epoch_{epoch}_batch_{batch_idx}_sample_{i}.png")
            
            print(f"  Generated {len(images)} sample images")
            
    except Exception as e:
        print(f"Error generating samples: {e}")
    finally:
        trainer.imagen.train()

def main():
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
    
    print(f"Using configuration: {args.config}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    
    # Setup datasets
    train_loader, val_loader = setup_datasets(config, use_synthetic=args.synthetic)
    
    # Setup model
    imagen = setup_model(config)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = TryOnImagenTrainer(
        imagen=imagen,
        lr=config['learning_rate'],
        eps=config['eps'],
        betas=config['betas'],
        max_grad_norm=config['max_grad_norm'],
        checkpoint_every=config['checkpoint_every'],
        checkpoint_path=config['checkpoint_path'],
        use_ema=config['use_ema'],
        fp16=config['use_fp16'],
        accelerate_gradient_accumulation_steps=config['gradient_accumulation_steps'],
        warmup_steps=config['warmup_steps'],
        cosine_decay_max_steps=config.get('cosine_decay_max_steps'),
    )
    
    trainer.add_train_dataloader(train_loader)
    trainer.add_valid_dataloader(val_loader)
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_path'], exist_ok=True)
    
    # Resume from checkpoint if specified
    if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
        trainer.load(config['resume_from_checkpoint'])
        print(f"Resumed from checkpoint: {config['resume_from_checkpoint']}")
    
    # Training stages
    if args.unet_stage == 1:
        # Stage 1: Train Base UNet
        train_stage(trainer, unet_number=1, 
                   num_epochs=config['train_base_unet_epochs'], 
                   config=config,
                   stage_name="base_unet")
    elif args.unet_stage == 2:
        # Stage 2: Train SR UNet  
        train_stage(trainer, unet_number=2, 
                   num_epochs=config['train_sr_unet_epochs'], 
                   config=config,
                   stage_name="sr_unet")
    
    print(f"\\nTraining completed for UNet {args.unet_stage}!")

if __name__ == "__main__":
    main()