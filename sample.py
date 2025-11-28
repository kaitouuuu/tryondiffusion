import os
import sys
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tryondiffusion import TryOnImagen, get_unet_by_name
from datasets.tryon_dataset import TryOnDataset
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from trained TryOnDiffusion model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--ca-image', type=str, required=True,
                       help='Path to clothing-agnostic person image')
    parser.add_argument('--garment-image', type=str, required=True,
                       help='Path to garment image')
    parser.add_argument('--person-pose', type=str, required=True,
                       help='Path to person pose JSON file')
    parser.add_argument('--garment-pose', type=str, required=True,
                       help='Path to garment pose JSON file')
    parser.add_argument('--output', type=str, default='output.png',
                       help='Output image path')
    parser.add_argument('--cond-scale', type=float, default=3.0,
                       help='Conditioning scale for classifier-free guidance')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of sampling steps (default: use model default)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    return parser.parse_args()

def load_and_preprocess_image(image_path, image_size=(256, 256)):
    """Load and preprocess an image"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_pose(pose_path):
    """Load pose keypoints from JSON file"""
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)
    
    # Extract keypoints - handle different formats
    if 'keypoints' in pose_data:
        keypoints = pose_data['keypoints']
    elif 'pose_keypoints_2d' in pose_data:  # OpenPose format
        keypoints = pose_data['pose_keypoints_2d']
    else:
        # Fallback: assume the JSON contains direct keypoint coordinates
        keypoints = pose_data
        
    # Convert to tensor and reshape to (18, 2) for x,y coordinates
    if len(keypoints) == 54:  # 18 keypoints * 3 (x,y,confidence)
        # Extract only x,y coordinates, skip confidence scores
        xy_coords = []
        for i in range(0, len(keypoints), 3):
            xy_coords.extend([keypoints[i], keypoints[i+1]])
        keypoints = xy_coords
        
    pose_tensor = torch.tensor(keypoints).reshape(18, 2).float()
    
    # Normalize coordinates to [0, 1] if they appear to be in pixel coordinates
    if pose_tensor.max() > 2:
        pose_tensor[:, 0] /= 512  # Assuming max width of 512
        pose_tensor[:, 1] /= 512  # Assuming max height of 512
        
    return pose_tensor.unsqueeze(0)  # Add batch dimension

def load_trained_model(checkpoint_path):
    """Load a trained TryOnImagen model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create UNets
    base_unet = get_unet_by_name("base")
    sr_unet = get_unet_by_name("sr")
    
    # Create Imagen model
    imagen = TryOnImagen(
        unets=(base_unet, sr_unet),
        image_sizes=((128, 128), (256, 256)),
        timesteps=(1000, 250),
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'imagen_state_dict' in checkpoint:
        imagen.load_state_dict(checkpoint['imagen_state_dict'])
    elif 'model_state_dict' in checkpoint:
        imagen.load_state_dict(checkpoint['model_state_dict'])
    elif 'imagen' in checkpoint:
        imagen.load_state_dict(checkpoint['imagen'])
    else:
        # Assume the checkpoint is the state dict directly
        imagen.load_state_dict(checkpoint)
    
    imagen.eval()
    print("Model loaded successfully!")
    return imagen

def sample_images(imagen, ca_image, garment_image, person_pose, garment_pose, 
                 cond_scale=3.0, steps=None, seed=42):
    """Generate images using the trained model"""
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print("Generating images...")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imagen = imagen.to(device)
    ca_image = ca_image.to(device)
    garment_image = garment_image.to(device)
    person_pose = person_pose.to(device)
    garment_pose = garment_pose.to(device)
    
    with torch.no_grad():
        # Prepare sampling arguments
        sample_kwargs = {
            'ca_images': ca_image,
            'garment_images': garment_image,
            'person_poses': person_pose,
            'garment_poses': garment_pose,
            'batch_size': 1,
            'cond_scale': cond_scale,
            'start_at_unet_number': 1,
            'return_all_unet_outputs': True,
            'return_pil_images': True,
            'use_tqdm': True,
            'use_one_unet_in_gpu': True,
        }
        
        # Add custom timesteps if specified
        if steps is not None:
            sample_kwargs['timesteps'] = (steps, steps // 4)
        
        # Generate images
        images = imagen.sample(**sample_kwargs)
        
    return images

def main():
    args = parse_args()
    
    # Check if files exist
    for file_path in [args.checkpoint, args.ca_image, args.garment_image, 
                     args.person_pose, args.garment_pose]:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return
    
    # Load model
    imagen = load_trained_model(args.checkpoint)
    
    # Load and preprocess inputs
    print("Loading and preprocessing inputs...")
    ca_image = load_and_preprocess_image(args.ca_image)
    garment_image = load_and_preprocess_image(args.garment_image)
    person_pose = load_pose(args.person_pose)
    garment_pose = load_pose(args.garment_pose)
    
    print(f"CA image shape: {ca_image.shape}")
    print(f"Garment image shape: {garment_image.shape}")
    print(f"Person pose shape: {person_pose.shape}")
    print(f"Garment pose shape: {garment_pose.shape}")
    
    # Generate images
    try:
        images = sample_images(
            imagen, ca_image, garment_image, person_pose, garment_pose,
            cond_scale=args.cond_scale, steps=args.steps, seed=args.seed
        )
        
        # Save results
        print(f"Saving results...")
        
        # images is a list of lists: [base_unet_outputs, sr_unet_outputs]
        # We want the final high-resolution result
        if len(images) >= 2 and len(images[1]) > 0:
            # Save the super-resolution output
            final_image = images[1][0]  # First image from SR UNet
            final_image.save(args.output)
            print(f"Final result saved to: {args.output}")
            
            # Also save intermediate result from base UNet
            base_output = os.path.splitext(args.output)[0] + "_base.png"
            if len(images[0]) > 0:
                images[0][0].save(base_output)
                print(f"Base UNet result saved to: {base_output}")
        else:
            # Fallback: save whatever we got
            if len(images) > 0 and len(images[0]) > 0:
                images[0][0].save(args.output)
                print(f"Result saved to: {args.output}")
            else:
                print("Error: No images generated")
                return
                
        print("Sampling completed successfully!")
        
    except Exception as e:
        print(f"Error during sampling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()