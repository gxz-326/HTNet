import os
import pandas as pd
import shutil
import argparse
from sklearn.model_selection import train_test_split


def prepare_fnp_dataset(data_root, output_csv, split_ratio=[0.7, 0.15, 0.15]):
    """
    Prepare FNP dataset and create annotation CSV file
    
    Expected input structure:
    data_root/
        grade_0/
            img1.jpg
            img2.jpg
        grade_1/
        ...
        grade_5/
    
    Output CSV columns: image_path, grade, side, split
    """
    print(f'Preparing FNP dataset from {data_root}')
    
    data = []
    for grade_folder in sorted(os.listdir(data_root)):
        if grade_folder.startswith('grade_'):
            grade = int(grade_folder.split('_')[1])
            grade_path = os.path.join(data_root, grade_folder)
            
            if os.path.isdir(grade_path):
                for img_file in os.listdir(grade_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(grade_path, img_file)
                        
                        side = 'unknown'
                        if 'left' in img_file.lower():
                            side = 'left'
                        elif 'right' in img_file.lower():
                            side = 'right'
                        
                        data.append({
                            'image_path': img_path,
                            'grade': grade,
                            'side': side
                        })
    
    df = pd.DataFrame(data)
    print(f'Total images found: {len(df)}')
    
    print('\nGrade distribution:')
    print(df['grade'].value_counts().sort_index())
    
    train_ratio, val_ratio, test_ratio = split_ratio
    
    train_df, temp_df = train_test_split(df, test_size=(1-train_ratio), 
                                         stratify=df['grade'], random_state=42)
    val_df, test_df = train_test_split(temp_df, 
                                       test_size=test_ratio/(val_ratio+test_ratio),
                                       stratify=temp_df['grade'], random_state=42)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    final_df.to_csv(output_csv, index=False)
    print(f'\nDataset annotation saved to {output_csv}')
    print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
    
    return final_df


def prepare_ckplus_dataset(data_root, output_root, grade_mapping_file=None):
    """
    Prepare CK+ dataset for facial palsy task
    
    CK+ is originally for facial expressions, so we need to adapt it.
    This function organizes CK+ images into grade folders based on expression intensity.
    
    Input structure:
    data_root/
        S005/
            001/
                S005_001_00000001.png
                ...
    
    Output structure:
    output_root/
        grade_0/
        grade_1/
        ...
    """
    print(f'Preparing CK+ dataset from {data_root}')
    
    os.makedirs(output_root, exist_ok=True)
    
    grade_mapping = {}
    if grade_mapping_file and os.path.exists(grade_mapping_file):
        df = pd.read_csv(grade_mapping_file)
        for _, row in df.iterrows():
            key = f"{row['subject']}_{row['sequence']}"
            grade_mapping[key] = row['grade']
    
    for grade in range(6):
        os.makedirs(os.path.join(output_root, f'grade_{grade}'), exist_ok=True)
    
    total_copied = 0
    for subject in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject)
        if os.path.isdir(subject_path):
            for sequence in os.listdir(subject_path):
                sequence_path = os.path.join(subject_path, sequence)
                if os.path.isdir(sequence_path):
                    images = sorted([f for f in os.listdir(sequence_path) 
                                   if f.endswith('.png')])
                    
                    if len(images) > 0:
                        peak_img = images[-1]
                        
                        key = f"{subject}_{sequence}"
                        grade = grade_mapping.get(key, 0)
                        
                        src_path = os.path.join(sequence_path, peak_img)
                        dst_path = os.path.join(output_root, f'grade_{grade}', 
                                              f'{subject}_{sequence}_{peak_img}')
                        shutil.copy2(src_path, dst_path)
                        total_copied += 1
    
    print(f'Copied {total_copied} images to {output_root}')
    
    data = []
    for grade in range(6):
        grade_path = os.path.join(output_root, f'grade_{grade}')
        count = len([f for f in os.listdir(grade_path) if f.endswith('.png')])
        data.append({'grade': grade, 'count': count})
        print(f'Grade {grade}: {count} images')
    
    return pd.DataFrame(data)


def convert_3class_to_6class(csv_file, output_csv):
    """
    Convert 3-class annotation to 6-class (House-Brackmann scale)
    
    Mapping example:
    Class 0 (Normal) -> Grade 1 (Normal function)
    Class 1 (Mild) -> Grades 2-3
    Class 2 (Severe) -> Grades 4-6
    """
    df = pd.read_csv(csv_file)
    
    def map_class_to_grade(cls):
        if cls == 0:
            return 1
        elif cls == 1:
            return 2
        else:
            return 4
    
    df['grade'] = df['class'].apply(map_class_to_grade)
    df.to_csv(output_csv, index=False)
    print(f'Converted annotation saved to {output_csv}')
    print('Grade distribution:')
    print(df['grade'].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for facial palsy recognition')
    
    parser.add_argument('--dataset_type', type=str, required=True, 
                       choices=['FNP', 'CK+', 'convert'],
                       help='Type of dataset to prepare')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the input dataset')
    parser.add_argument('--output_csv', type=str, default='dataset_annotation.csv',
                       help='Output CSV file path')
    parser.add_argument('--output_root', type=str, default='./datasets/prepared',
                       help='Output root directory (for CK+ preparation)')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                       help='Train/Val/Test split ratio')
    parser.add_argument('--grade_mapping', type=str, default=None,
                       help='Grade mapping CSV file for CK+ dataset')
    
    args = parser.parse_args()
    
    if args.dataset_type == 'FNP':
        prepare_fnp_dataset(args.data_root, args.output_csv, args.split_ratio)
    
    elif args.dataset_type == 'CK+':
        prepare_ckplus_dataset(args.data_root, args.output_root, args.grade_mapping)
        prepare_fnp_dataset(args.output_root, args.output_csv, args.split_ratio)
    
    elif args.dataset_type == 'convert':
        input_csv = args.data_root
        convert_3class_to_6class(input_csv, args.output_csv)
    
    print('\nDataset preparation completed!')


if __name__ == '__main__':
    main()
