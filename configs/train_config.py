# Training Configuration for TryOnDiffusion

TRAIN_CONFIG = {
    # Dataset paths
    'dataset_root': 'datasets/',
    'train_csv': 'datasets/metadata/train.csv',
    'val_csv': 'datasets/metadata/val.csv',
    
    # Model parameters
    'base_image_size': (128, 128),
    'sr_image_size': (256, 256),
    'timesteps': (1000, 250),  # (Base UNet timesteps, SR UNet timesteps)
    
    # Training parameters
    'batch_size': 4,  # Adjust based on GPU memory (reduce if OOM)
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'gradient_accumulation_steps': 4,  # Effective batch size = batch_size * gradient_accumulation_steps
    'max_grad_norm': 1.0,
    'warmup_steps': 1000,
    
    # Training stages
    'train_base_unet_epochs': 50,    # Train base UNet first
    'train_sr_unet_epochs': 50,      # Then train SR UNet
    
    # Optimizer settings
    'betas': (0.9, 0.99),
    'eps': 1e-8,
    'weight_decay': 0.01,
    
    # Checkpointing
    'checkpoint_every': 1000,  # Save checkpoint every N steps
    'checkpoint_path': './checkpoints',
    'resume_from_checkpoint': None,  # Set to checkpoint path to resume training
    'save_every_n_epochs': 5,
    
    # Validation
    'validate_every': 2000,  # Validate every N steps
    'num_validation_samples': 4,
    
    # Hardware optimization
    'use_fp16': True,  # Mixed precision training
    'use_ema': True,   # Exponential moving average
    'num_workers': 4,  # DataLoader workers
    'pin_memory': True,
    
    # Scheduler
    'use_cosine_scheduler': True,
    'cosine_decay_max_steps': None,  # Will be set based on total training steps
    
    # Conditioning
    'cond_scale': 3.0,  # Classifier-free guidance scale for validation sampling
    'unconditional_prob': 0.1,  # Probability of unconditional training
    
    # Logging
    'log_every': 100,
    'sample_every': 5000,  # Generate sample images every N steps
    'num_sample_images': 4,
    
    # Advanced settings
    'gradient_checkpointing': False,  # Enable to save memory at cost of speed
    'compile_model': False,  # PyTorch 2.0 compilation (if available)
}

# Hardware-specific configs
HARDWARE_CONFIGS = {
    # For GPUs with 8GB VRAM
    'low_memory': {
        'batch_size': 1,
        'gradient_accumulation_steps': 8,
        'use_fp16': True,
        'gradient_checkpointing': True,
        'num_workers': 2,
    },
    
    # For GPUs with 16GB VRAM
    'medium_memory': {
        'batch_size': 2,
        'gradient_accumulation_steps': 4,
        'use_fp16': True,
        'gradient_checkpointing': False,
        'num_workers': 4,
    },
    
    # For GPUs with 24GB+ VRAM
    'high_memory': {
        'batch_size': 4,
        'gradient_accumulation_steps': 2,
        'use_fp16': False,
        'gradient_checkpointing': False,
        'num_workers': 8,
    }
}

def get_config(memory_config='medium_memory'):
    """
    Get training config merged with hardware-specific settings.
    
    Args:
        memory_config: 'low_memory', 'medium_memory', or 'high_memory'
    """
    config = TRAIN_CONFIG.copy()
    if memory_config in HARDWARE_CONFIGS:
        config.update(HARDWARE_CONFIGS[memory_config])
    return config