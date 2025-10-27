import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import random


class FacialPalsyAugmentation:
    """
    Data augmentation specifically designed for facial palsy images
    Preserves facial asymmetry which is crucial for grading
    """
    def __init__(self, image_size=224, is_training=True):
        self.image_size = image_size
        self.is_training = is_training
        
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, image):
        """Apply augmentation to image tensor"""
        if isinstance(image, torch.Tensor):
            image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        if self.is_training:
            image = self.add_noise(image)
            image = self.adjust_lighting(image)
        
        image = self.transform(image)
        return image
    
    def add_noise(self, image, noise_level=0.02):
        """Add Gaussian noise to simulate different imaging conditions"""
        if random.random() < 0.5:
            noise = np.random.randn(*image.shape) * noise_level * 255
            noisy_image = image.astype(np.float32) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            return noisy_image
        return image
    
    def adjust_lighting(self, image, gamma_range=(0.8, 1.2)):
        """
        Adjust lighting using gamma correction
        Simulates different lighting conditions
        """
        if random.random() < 0.5:
            gamma = random.uniform(*gamma_range)
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype(np.uint8)
            return cv2.LUT(image, table)
        return image


class AsymmetryPreservingAugmentation:
    """
    Augmentation that preserves facial asymmetry
    Critical for facial palsy assessment
    """
    def __init__(self, image_size=224):
        self.image_size = image_size
    
    def __call__(self, image):
        """
        Apply augmentations that preserve left-right asymmetry
        """
        if isinstance(image, torch.Tensor):
            image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        image = self.random_crop_and_resize(image)
        image = self.random_scale(image)
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image
    
    def random_crop_and_resize(self, image, crop_ratio=0.9):
        """
        Random crop and resize while maintaining aspect ratio
        """
        if random.random() < 0.5:
            h, w = image.shape[:2]
            crop_h = int(h * crop_ratio)
            crop_w = int(w * crop_ratio)
            
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            
            cropped = image[top:top+crop_h, left:left+crop_w]
            resized = cv2.resize(cropped, (self.image_size, self.image_size))
            return resized
        
        return image
    
    def random_scale(self, image, scale_range=(0.95, 1.05)):
        """Apply random scaling"""
        if random.random() < 0.5:
            scale = random.uniform(*scale_range)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            scaled = cv2.resize(image, (new_w, new_h))
            
            if scale > 1.0:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                return scaled[start_h:start_h+h, start_w:start_w+w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                padded = cv2.copyMakeBorder(
                    scaled, pad_h, h - new_h - pad_h,
                    pad_w, w - new_w - pad_w,
                    cv2.BORDER_REPLICATE
                )
                return padded
        
        return image


class MixUp:
    """
    MixUp augmentation for facial palsy images
    Helps model learn smoother decision boundaries between grades
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        """
        Apply MixUp augmentation to a batch
        
        Args:
            batch_images: Tensor of shape (batch_size, C, H, W)
            batch_labels: Tensor of shape (batch_size,)
        
        Returns:
            mixed_images, mixed_labels_a, mixed_labels_b, lam
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        labels_a = batch_labels
        labels_b = batch_labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def get_augmentation_pipeline(augmentation_type='standard', image_size=224, is_training=True):
    """
    Get augmentation pipeline based on type
    
    Args:
        augmentation_type: 'standard', 'asymmetry_preserving', or 'none'
        image_size: Target image size
        is_training: Whether in training mode
    
    Returns:
        Augmentation callable
    """
    if not is_training or augmentation_type == 'none':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    elif augmentation_type == 'standard':
        return FacialPalsyAugmentation(image_size=image_size, is_training=True)
    
    elif augmentation_type == 'asymmetry_preserving':
        return AsymmetryPreservingAugmentation(image_size=image_size)
    
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


class CutOut:
    """
    CutOut augmentation - randomly mask out regions
    Helps model become robust to occlusions
    """
    def __init__(self, n_holes=1, length=20):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, image):
        """
        Apply cutout to image tensor
        
        Args:
            image: Tensor of shape (C, H, W)
        """
        h, w = image.shape[1], image.shape[2]
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask).unsqueeze(0)
        image = image * mask
        
        return image


class RandomErasing:
    """
    Randomly erase rectangular regions in the image
    Similar to CutOut but with variable size
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, image):
        if random.random() > self.probability:
            return image
        
        _, h, w = image.shape
        area = h * w
        
        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            
            h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
            w_erase = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if h_erase < h and w_erase < w:
                x1 = random.randint(0, h - h_erase)
                y1 = random.randint(0, w - w_erase)
                
                image[:, x1:x1+h_erase, y1:y1+w_erase] = 0
                return image
        
        return image


if __name__ == '__main__':
    print("Data Augmentation Module for Facial Palsy Recognition")
    print("=" * 60)
    
    augmentation_types = ['standard', 'asymmetry_preserving', 'none']
    
    for aug_type in augmentation_types:
        aug = get_augmentation_pipeline(aug_type, image_size=224, is_training=True)
        print(f"\n{aug_type} augmentation initialized successfully")
    
    mixup = MixUp(alpha=0.2)
    print("\nMixUp augmentation initialized")
    
    cutout = CutOut(n_holes=1, length=20)
    print("CutOut augmentation initialized")
    
    random_erasing = RandomErasing(probability=0.5)
    print("RandomErasing augmentation initialized")
    
    print("\n" + "=" * 60)
    print("All augmentation modules ready!")
