import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TryOnDataset(Dataset):
    def __init__(self, csv_file, dataset_root, image_size=(128, 128), transform=None):
        """
        TryOn Dataset for loading person, garment, and pose data.
        
        Args:
            csv_file (str): Path to CSV file with metadata
            dataset_root (str): Root directory of dataset
            image_size (tuple): Target image size (height, width)
            transform: Image transforms to apply
        """
        self.df = pd.read_csv(csv_file)
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.transform = transform or self.get_default_transform()
        
    def get_default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.df)
    
    def load_pose(self, pose_path):
        """
        Load pose keypoints from JSON file.
        Expected format: COCO pose format with 18 keypoints.
        """
        full_path = os.path.join(self.dataset_root, pose_path)
        with open(full_path, 'r') as f:
            pose_data = json.load(f)
        
        # Extract keypoints - assuming COCO format
        if 'keypoints' in pose_data:
            keypoints = pose_data['keypoints']
        elif 'pose_keypoints_2d' in pose_data:  # OpenPose format
            keypoints = pose_data['pose_keypoints_2d']
        else:
            # Fallback: assume the JSON contains direct keypoint coordinates
            keypoints = pose_data
            
        # Convert to tensor and reshape to (18, 2) for x,y coordinates
        # If confidence scores are included, take only x,y coordinates
        if len(keypoints) == 54:  # 18 keypoints * 3 (x,y,confidence)
            keypoints = keypoints[::3] + keypoints[1::3]  # Take x,y coordinates only
            
        pose_tensor = torch.tensor(keypoints).reshape(18, 2).float()
        
        # Normalize coordinates to [0, 1] if they appear to be in pixel coordinates
        if pose_tensor.max() > 2:
            pose_tensor[:, 0] /= 512  # Assuming max width of 512
            pose_tensor[:, 1] /= 512  # Assuming max height of 512
            
        return pose_tensor
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        row = self.df.iloc[idx]
        
        try:
            # Load images
            person_img_path = os.path.join(self.dataset_root, row['person_image'])
            ca_img_path = os.path.join(self.dataset_root, row['ca_image'])
            garment_img_path = os.path.join(self.dataset_root, row['garment_image'])
            
            person_img = Image.open(person_img_path).convert('RGB')
            ca_img = Image.open(ca_img_path).convert('RGB')
            garment_img = Image.open(garment_img_path).convert('RGB')
            
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
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample in case of error
            dummy_img = torch.zeros(3, *self.image_size)
            dummy_pose = torch.zeros(18, 2)
            return {
                'person_images': dummy_img,
                'ca_images': dummy_img,
                'garment_images': dummy_img,
                'person_poses': dummy_pose,
                'garment_poses': dummy_pose
            }


def collate_fn(batch):
    """
    Custom collate function to batch the data properly.
    """
    return {
        'person_images': torch.stack([item['person_images'] for item in batch]),
        'ca_images': torch.stack([item['ca_images'] for item in batch]),
        'garment_images': torch.stack([item['garment_images'] for item in batch]),
        'person_poses': torch.stack([item['person_poses'] for item in batch]),
        'garment_poses': torch.stack([item['garment_poses'] for item in batch]),
    }


class SyntheticTryOnDataset(Dataset):
    """
    Synthetic dataset for testing purposes.
    Generates random data that matches the expected format.
    """
    def __init__(self, num_samples=1000, image_size=(128, 128)):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random synthetic data
        return {
            'person_images': torch.randn(3, *self.image_size),
            'ca_images': torch.randn(3, *self.image_size),
            'garment_images': torch.randn(3, *self.image_size),
            'person_poses': torch.randn(18, 2),
            'garment_poses': torch.randn(18, 2),
        }