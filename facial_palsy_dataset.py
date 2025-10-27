import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN


class FacialPalsyDataset(Dataset):
    """
    Dataset for Facial Palsy Recognition and Grading
    Supports FNP (Facial Nerve Palsy) and CK+ datasets
    """
    def __init__(self, data_root, csv_file=None, dataset_type='FNP', 
                 image_size=224, transform=None, split='train'):
        """
        Args:
            data_root: Root directory of the dataset
            csv_file: CSV file with annotations (image_path, grade, side)
            dataset_type: 'FNP' or 'CK+'
            image_size: Size to resize images to
            transform: Optional transform to be applied on a sample
            split: 'train', 'val', or 'test'
        """
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.transform = transform
        self.split = split
        
        self.mtcnn = MTCNN(margin=0, image_size=image_size, 
                          select_largest=True, post_process=False, 
                          device='cuda' if torch.cuda.is_available() else 'cpu')
        
        if csv_file and os.path.exists(csv_file):
            self.data_df = pd.read_csv(csv_file)
            self.data_df = self.data_df[self.data_df['split'] == split]
        else:
            self.data_df = self._create_dataframe_from_directory()
        
        self.image_paths = self.data_df['image_path'].tolist()
        self.labels = self.data_df['grade'].tolist()
        
    def _create_dataframe_from_directory(self):
        """
        Create dataframe from directory structure
        Expected structure:
        data_root/
            grade_0/
                img1.jpg
                img2.jpg
            grade_1/
            ...
        """
        data = []
        if os.path.exists(self.data_root):
            for grade_folder in sorted(os.listdir(self.data_root)):
                grade_path = os.path.join(self.data_root, grade_folder)
                if os.path.isdir(grade_path) and grade_folder.startswith('grade_'):
                    grade = int(grade_folder.split('_')[1])
                    for img_file in os.listdir(grade_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(grade_path, img_file)
                            data.append({
                                'image_path': img_path,
                                'grade': grade,
                                'split': self.split
                            })
        return pd.DataFrame(data)
    
    def _detect_facial_landmarks(self, image):
        """Detect 5 facial landmarks: left_eye, right_eye, nose, left_mouth, right_mouth"""
        batch_boxes, _, batch_landmarks = self.mtcnn.detect(image, landmarks=True)
        
        if batch_landmarks is None:
            h, w = image.shape[:2]
            scale_h, scale_w = h / 28, w / 28
            default_landmarks = np.array([[[9.528073 * scale_w, 11.062551 * scale_h],
                                          [21.396168 * scale_w, 10.919773 * scale_h],
                                          [15.380184 * scale_w, 17.380562 * scale_h],
                                          [10.255435 * scale_w, 22.121233 * scale_h],
                                          [20.583706 * scale_w, 22.25584 * scale_h]]])
            return default_landmarks[0].astype(int)
        
        return batch_landmarks[0].astype(int)
    
    def _extract_facial_regions(self, image, landmarks, region_size=14):
        """
        Extract 5 facial regions based on landmarks
        Returns: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        regions = []
        h, w = image.shape[:2]
        
        for landmark in landmarks:
            x, y = landmark
            x = np.clip(x, region_size, w - region_size)
            y = np.clip(y, region_size, h - region_size)
            
            region = image[y-region_size:y+region_size, 
                          x-region_size:x+region_size]
            
            if region.shape[0] != region_size * 2 or region.shape[1] != region_size * 2:
                region = cv2.resize(region, (region_size * 2, region_size * 2))
            
            regions.append(region)
        
        return regions
    
    def _create_hierarchical_input(self, regions):
        """
        Create hierarchical input similar to HTNet
        Combines regions: [left_eye, left_mouth] horizontally and [right_eye, right_mouth] horizontally
        Then stacks them vertically
        """
        left_eye, right_eye, nose, left_mouth, right_mouth = regions
        
        left_eye_mouth = cv2.hconcat([left_eye, left_mouth])
        right_eye_mouth = cv2.hconcat([right_eye, right_mouth])
        combined = cv2.vconcat([left_eye_mouth, right_eye_mouth])
        
        return combined
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        landmarks = self._detect_facial_landmarks(image)
        regions = self._extract_facial_regions(image, landmarks)
        hierarchical_input = self._create_hierarchical_input(regions)
        
        hierarchical_input = hierarchical_input.astype(np.float32) / 255.0
        hierarchical_input = torch.from_numpy(hierarchical_input).permute(2, 0, 1)
        
        if self.transform:
            hierarchical_input = self.transform(hierarchical_input)
        
        return hierarchical_input, torch.tensor(label, dtype=torch.long)


class CKPlusFacialPalsyDataset(Dataset):
    """
    CK+ Dataset adapter for Facial Palsy task
    Note: CK+ is originally for facial expressions, adapted for palsy grading
    """
    def __init__(self, data_root, annotation_file=None, image_size=224, 
                 transform=None, split='train'):
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        self.split = split
        
        self.mtcnn = MTCNN(margin=0, image_size=image_size,
                          select_largest=True, post_process=False,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
        
        self.image_paths, self.labels = self._load_ckplus_data()
    
    def _load_ckplus_data(self):
        """
        Load CK+ data
        Expected structure:
        data_root/
            S005/
                001/
                    S005_001_00000001.png
                    ...
        """
        image_paths = []
        labels = []
        
        if os.path.exists(self.data_root):
            for subject in sorted(os.listdir(self.data_root)):
                subject_path = os.path.join(self.data_root, subject)
                if os.path.isdir(subject_path):
                    for sequence in os.listdir(subject_path):
                        sequence_path = os.path.join(subject_path, sequence)
                        if os.path.isdir(sequence_path):
                            images = sorted([f for f in os.listdir(sequence_path) 
                                           if f.endswith('.png')])
                            if len(images) > 0:
                                img_path = os.path.join(sequence_path, images[-1])
                                image_paths.append(img_path)
                                label = self._get_palsy_grade_from_expression(images[-1])
                                labels.append(label)
        
        return image_paths, labels
    
    def _get_palsy_grade_from_expression(self, filename):
        """
        Map CK+ expressions to palsy grades
        This is a simplified mapping - adjust based on your needs
        """
        return 0
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


def create_facial_palsy_dataloader(data_root, batch_size=32, dataset_type='FNP',
                                   num_workers=4, csv_file=None, split='train'):
    """
    Create dataloader for facial palsy dataset
    """
    if dataset_type == 'FNP':
        dataset = FacialPalsyDataset(
            data_root=data_root,
            csv_file=csv_file,
            dataset_type=dataset_type,
            image_size=224,
            split=split
        )
    elif dataset_type == 'CK+':
        dataset = CKPlusFacialPalsyDataset(
            data_root=data_root,
            image_size=224,
            split=split
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
