import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from PIL import Image


class YFPFacialPalsyDataset(Dataset):
    """
    YouTube Facial Palsy (YFP) Dataset for facial palsy detection.
    Binary classification: 0 - Normal, 1 - Facial Palsy
    """
    
    def __init__(self, csv_file, data_root, image_size=28, transform=None, use_optical_flow=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
                               Expected columns: 'subject_id', 'image_path', 'label' (0: normal, 1: palsy)
            data_root (string): Directory with all the images.
            image_size (int): Size to resize images to.
            transform (callable, optional): Optional transform to be applied on a sample.
            use_optical_flow (bool): Whether to use optical flow features.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        self.use_optical_flow = use_optical_flow
        self.mtcnn = MTCNN(margin=0, image_size=image_size, select_largest=True, 
                          post_process=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.data_root, self.data_frame.iloc[idx]['image_path'])
        label = self.data_frame.iloc[idx]['label']
        
        # Load image
        image = cv2.imread(img_name)
        if image is None:
            # Return a zero tensor if image cannot be loaded
            return torch.zeros(3, self.image_size, self.image_size), label
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Detect and crop face
        try:
            face = self.mtcnn(image)
            if face is None:
                # Fallback: resize original image
                image_np = np.array(image)
                face_np = cv2.resize(image_np, (self.image_size, self.image_size))
                face = torch.from_numpy(face_np).permute(2, 0, 1).float()
            else:
                face = face.float()
        except:
            # Fallback: resize original image
            image_np = np.array(image)
            face_np = cv2.resize(image_np, (self.image_size, self.image_size))
            face = torch.from_numpy(face_np).permute(2, 0, 1).float()
        
        if self.transform:
            face = self.transform(face)
        
        return face, label
    
    def get_subject_ids(self):
        """Get unique subject IDs for cross-validation"""
        return self.data_frame['subject_id'].unique()
    
    def get_subject_data(self, subject_id):
        """Get data for a specific subject"""
        subject_df = self.data_frame[self.data_frame['subject_id'] == subject_id]
        return subject_df


class YFPOpticalFlowDataset(Dataset):
    """
    YFP Dataset with optical flow features for facial palsy detection.
    Uses onset and apex frames to compute optical flow.
    """
    
    def __init__(self, csv_file, data_root, image_size=28, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
                               Expected columns: 'subject_id', 'onset_frame', 'apex_frame', 'label'
            data_root (string): Directory with all the images.
            image_size (int): Size to resize images to.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        self.mtcnn = MTCNN(margin=0, image_size=image_size, select_largest=True, 
                          post_process=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def __len__(self):
        return len(self.data_frame)
    
    def compute_optical_flow(self, onset_path, apex_path):
        """Compute optical flow between onset and apex frames"""
        # Load images
        onset_img = cv2.imread(os.path.join(self.data_root, onset_path))
        apex_img = cv2.imread(os.path.join(self.data_root, apex_path))
        
        if onset_img is None or apex_img is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Convert to RGB for face detection
        onset_rgb = cv2.cvtColor(onset_img, cv2.COLOR_BGR2RGB)
        apex_rgb = cv2.cvtColor(apex_img, cv2.COLOR_BGR2RGB)
        
        onset_pil = Image.fromarray(onset_rgb)
        apex_pil = Image.fromarray(apex_rgb)
        
        # Detect and crop faces
        try:
            face_onset = self.mtcnn(onset_pil)
            face_apex = self.mtcnn(apex_pil)
            
            if face_onset is None or face_apex is None:
                face_onset_np = cv2.resize(onset_img, (self.image_size, self.image_size))
                face_apex_np = cv2.resize(apex_img, (self.image_size, self.image_size))
            else:
                face_onset_np = np.array(face_onset.permute(1, 2, 0).int().numpy()).astype('uint8')
                face_apex_np = np.array(face_apex.permute(1, 2, 0).int().numpy()).astype('uint8')
        except:
            face_onset_np = cv2.resize(onset_img, (self.image_size, self.image_size))
            face_apex_np = cv2.resize(apex_img, (self.image_size, self.image_size))
        
        # Convert to grayscale for optical flow
        onset_gray = cv2.cvtColor(face_onset_np, cv2.COLOR_RGB2GRAY)
        apex_gray = cv2.cvtColor(face_apex_np, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(onset_gray, apex_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert flow to magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize components
        u = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        v = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        magnitude = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create 3-channel optical flow image (u, v, magnitude)
        flow_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        flow_image[:, :, 0] = u
        flow_image[:, :, 1] = v
        flow_image[:, :, 2] = magnitude
        
        return flow_image
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        onset_path = self.data_frame.iloc[idx]['onset_frame']
        apex_path = self.data_frame.iloc[idx]['apex_frame']
        label = self.data_frame.iloc[idx]['label']
        
        # Compute optical flow
        flow_image = self.compute_optical_flow(onset_path, apex_path)
        
        # Convert to tensor
        flow_tensor = torch.from_numpy(flow_image).permute(2, 0, 1).float()
        
        if self.transform:
            flow_tensor = self.transform(flow_tensor)
        
        return flow_tensor, label
    
    def get_subject_ids(self):
        """Get unique subject IDs for cross-validation"""
        return self.data_frame['subject_id'].unique()
