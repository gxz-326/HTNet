#!/usr/bin/env python3
"""
Helper script to create a sample YFP dataset structure for testing.
This creates dummy images and CSV files to test the system.
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse


def create_sample_face_image(width=224, height=224, add_asymmetry=False):
    """
    Create a synthetic face image for testing.
    
    Args:
        width: Image width
        height: Image height
        add_asymmetry: If True, adds asymmetry to simulate facial palsy
    
    Returns:
        numpy array: RGB image
    """
    # Create blank image
    img = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Face outline (oval)
    center_x, center_y = width // 2, height // 2
    cv2.ellipse(img, (center_x, center_y), (width//3, height//2), 0, 0, 360, (150, 150, 150), -1)
    
    # Eyes
    left_eye_x = center_x - width // 6
    right_eye_x = center_x + width // 6
    eye_y = center_y - height // 6
    
    if add_asymmetry:
        # Asymmetric eyes (simulating palsy)
        cv2.circle(img, (left_eye_x, eye_y), width//20, (50, 50, 50), -1)
        cv2.circle(img, (right_eye_x, eye_y + 10), width//25, (50, 50, 50), -1)  # Drooping
        
        # Asymmetric mouth
        mouth_pts = np.array([
            [center_x - width//6, center_y + height//4],
            [center_x, center_y + height//4 + 15],  # Drooping on one side
            [center_x + width//6, center_y + height//4]
        ])
    else:
        # Symmetric eyes
        cv2.circle(img, (left_eye_x, eye_y), width//20, (50, 50, 50), -1)
        cv2.circle(img, (right_eye_x, eye_y), width//20, (50, 50, 50), -1)
        
        # Symmetric mouth
        mouth_pts = np.array([
            [center_x - width//6, center_y + height//4],
            [center_x, center_y + height//4 + 5],
            [center_x + width//6, center_y + height//4]
        ])
    
    # Draw mouth
    cv2.polylines(img, [mouth_pts], False, (50, 50, 50), 2)
    
    # Nose
    cv2.line(img, (center_x, center_y - 10), (center_x, center_y + 30), (100, 100, 100), 2)
    
    # Add some noise for realism
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def create_sample_dataset(output_dir='./datasets/YFP_sample', num_subjects=10, images_per_subject=5):
    """
    Create a sample YFP dataset for testing.
    
    Args:
        output_dir: Directory to save the sample dataset
        num_subjects: Number of subjects to create
        images_per_subject: Number of images per subject
    """
    print(f"Creating sample YFP dataset in: {output_dir}")
    print(f"Subjects: {num_subjects}, Images per subject: {images_per_subject}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store CSV data
    csv_data_raw = []
    csv_data_flow = []
    
    # Create subjects
    for subject_idx in range(num_subjects):
        subject_id = f"subject_{subject_idx+1:03d}"
        subject_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Randomly assign label (50% palsy, 50% normal)
        label = 1 if subject_idx % 2 == 0 else 0
        has_palsy = label == 1
        
        print(f"  Creating {subject_id} (Label: {'Palsy' if has_palsy else 'Normal'})")
        
        # Create images
        for img_idx in range(images_per_subject):
            # Create different variations for each image
            variation = img_idx / images_per_subject
            
            # Create onset (neutral) image
            onset_img = create_sample_face_image(add_asymmetry=has_palsy)
            onset_filename = f"onset_{img_idx+1:03d}.jpg"
            onset_path = os.path.join(subject_dir, onset_filename)
            cv2.imwrite(onset_path, cv2.cvtColor(onset_img, cv2.COLOR_RGB2BGR))
            
            # Create apex (peak expression) image with more pronounced features
            apex_img = create_sample_face_image(add_asymmetry=has_palsy)
            # Add more variation to apex
            if has_palsy:
                # Enhance asymmetry for palsy cases
                apex_img = cv2.addWeighted(apex_img, 1.0, onset_img, 0.3, 0)
            apex_filename = f"apex_{img_idx+1:03d}.jpg"
            apex_path = os.path.join(subject_dir, apex_filename)
            cv2.imwrite(apex_path, cv2.cvtColor(apex_img, cv2.COLOR_RGB2BGR))
            
            # Create regular image
            img = create_sample_face_image(add_asymmetry=has_palsy)
            img_filename = f"img_{img_idx+1:03d}.jpg"
            img_path = os.path.join(subject_dir, img_filename)
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Add to CSV data (raw images)
            csv_data_raw.append({
                'subject_id': subject_id,
                'image_path': f"{subject_id}/{img_filename}",
                'label': label
            })
            
            # Add to CSV data (optical flow)
            csv_data_flow.append({
                'subject_id': subject_id,
                'onset_frame': f"{subject_id}/{onset_filename}",
                'apex_frame': f"{subject_id}/{apex_filename}",
                'label': label
            })
    
    # Save CSV files
    csv_raw_path = os.path.join(output_dir, 'yfp_dataset.csv')
    df_raw = pd.DataFrame(csv_data_raw)
    df_raw.to_csv(csv_raw_path, index=False)
    print(f"\nSaved raw images CSV to: {csv_raw_path}")
    
    csv_flow_path = os.path.join(output_dir, 'yfp_optical_flow_dataset.csv')
    df_flow = pd.DataFrame(csv_data_flow)
    df_flow.to_csv(csv_flow_path, index=False)
    print(f"Saved optical flow CSV to: {csv_flow_path}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(csv_data_raw)}")
    print(f"  Normal samples: {sum(1 for d in csv_data_raw if d['label'] == 0)}")
    print(f"  Palsy samples: {sum(1 for d in csv_data_raw if d['label'] == 1)}")
    print(f"  Subjects: {num_subjects}")
    
    print("\nSample dataset created successfully!")
    print("\nTo train on this sample dataset, run:")
    print(f"\npython train_yfp_palsy_detection.py \\")
    print(f"    --train true \\")
    print(f"    --csv_file {csv_flow_path} \\")
    print(f"    --data_root {output_dir} \\")
    print(f"    --use_optical_flow true \\")
    print(f"    --epochs 20 \\")
    print(f"    --batch_size 8")


def main():
    parser = argparse.ArgumentParser(description='Create sample YFP dataset for testing')
    parser.add_argument('--output_dir', type=str, default='./datasets/YFP_sample',
                       help='Directory to save the sample dataset')
    parser.add_argument('--num_subjects', type=int, default=10,
                       help='Number of subjects to create')
    parser.add_argument('--images_per_subject', type=int, default=5,
                       help='Number of images per subject')
    
    args = parser.parse_args()
    
    create_sample_dataset(
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        images_per_subject=args.images_per_subject
    )


if __name__ == '__main__':
    main()
