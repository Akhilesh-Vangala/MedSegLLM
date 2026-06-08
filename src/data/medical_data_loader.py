import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional, Tuple
import cv2
from PIL import Image
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import ndimage
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


class MedicalImageDataset(Dataset):
    def __init__(self, 
                 image_paths: List[str],
                 mask_paths: Optional[List[str]] = None,
                 metadata: Optional[pd.DataFrame] = None,
                 transform=None,
                 target_size: Tuple[int, int] = (512, 512),
                 normalize: bool = True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.metadata = metadata
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
            image = self._load_nifti(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.zeros(self.target_size, dtype=np.uint8)
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        image = cv2.resize(image, self.target_size)
        
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        sample = {
            'image': torch.FloatTensor(image).permute(2, 0, 1),
            'image_path': image_path
        }
        
        if self.mask_paths:
            mask_path = self.mask_paths[idx]
            if mask_path and Path(mask_path).exists():
                if mask_path.endswith('.nii') or mask_path.endswith('.nii.gz'):
                    mask = self._load_nifti(mask_path)
                else:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        mask = np.zeros(self.target_size, dtype=np.uint8)
                
                mask = cv2.resize(mask, self.target_size)
                mask = (mask > 127).astype(np.int64)
                sample['mask'] = torch.LongTensor(mask)
        
        if self.metadata is not None:
            sample['metadata'] = self.metadata.iloc[idx].to_dict()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_nifti(self, path: str) -> np.ndarray:
        try:
            nii = nib.load(path)
            data = nii.get_fdata()
            if len(data.shape) == 3:
                data = data[:, :, data.shape[2] // 2]
            
            data = np.clip(data, 0, 255)
            return data.astype(np.uint8)
        except Exception as e:
            logger.warning(f"Error loading NIfTI {path}: {e}")
            return np.zeros(self.target_size, dtype=np.uint8)


class MedicalDataManager:
    def __init__(self, config: Dict):
        self.config = config
        self.image_paths = []
        self.mask_paths = []
        self.metadata = None
    
    def load_from_directory(self, 
                           image_dir: str,
                           mask_dir: Optional[str] = None,
                           metadata_file: Optional[str] = None) -> Tuple[List[str], Optional[List[str]]]:
        image_dir = Path(image_dir)
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.nii', '.nii.gz']
        self.image_paths = []
        
        for ext in image_extensions:
            self.image_paths.extend(list(image_dir.glob(f'*{ext}')))
            self.image_paths.extend(list(image_dir.glob(f'**/*{ext}')))
        
        self.image_paths = [str(p) for p in self.image_paths]
        
        if mask_dir:
            mask_dir = Path(mask_dir)
            self.mask_paths = []
            for ext in image_extensions:
                self.mask_paths.extend(list(mask_dir.glob(f'*{ext}')))
                self.mask_paths.extend(list(mask_dir.glob(f'**/*{ext}')))
            self.mask_paths = [str(p) for p in self.mask_paths]
            
            if len(self.mask_paths) != len(self.image_paths):
                logger.warning(f"Image count ({len(self.image_paths)}) != Mask count ({len(self.mask_paths)})")
                self.mask_paths = self.mask_paths[:len(self.image_paths)]
        else:
            self.mask_paths = None
        
        if metadata_file and Path(metadata_file).exists():
            self.metadata = pd.read_csv(metadata_file)
        
        logger.info(f"Loaded {len(self.image_paths)} images")
        return self.image_paths, self.mask_paths
    
    def create_dataloaders(self,
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          batch_size: int = 4,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if not self.image_paths:
            raise ValueError("No images loaded. Call load_from_directory first.")
        
        indices = list(range(len(self.image_paths)))
        
        train_idx, temp_idx = train_test_split(indices, test_size=test_size + val_size, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (test_size + val_size), random_state=42)
        
        train_images = [self.image_paths[i] for i in train_idx]
        train_masks = [self.mask_paths[i] for i in train_idx] if self.mask_paths else None
        
        val_images = [self.image_paths[i] for i in val_idx]
        val_masks = [self.mask_paths[i] for i in val_idx] if self.mask_paths else None
        
        test_images = [self.image_paths[i] for i in test_idx]
        test_masks = [self.mask_paths[i] for i in test_idx] if self.mask_paths else None
        
        train_metadata = self.metadata.iloc[train_idx] if self.metadata is not None else None
        val_metadata = self.metadata.iloc[val_idx] if self.metadata is not None else None
        test_metadata = self.metadata.iloc[test_idx] if self.metadata is not None else None
        
        train_dataset = MedicalImageDataset(train_images, train_masks, train_metadata)
        val_dataset = MedicalImageDataset(val_images, val_masks, val_metadata)
        test_dataset = MedicalImageDataset(test_images, test_masks, test_metadata)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
