import os
import argparse
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from Model import HTNet


class FacialPalsyPredictor:
    """
    Real-time facial palsy grading predictor
    """
    def __init__(self, model_path, num_classes=6, image_size=224, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.image_size = image_size
        
        print(f'Loading model from {model_path}')
        self.model = HTNet(
            image_size=image_size,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=num_classes
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.mtcnn = MTCNN(
            margin=0, 
            image_size=image_size,
            select_largest=True, 
            post_process=False,
            device=self.device
        )
        
        self.grade_names = {
            0: "Grade I - Normal",
            1: "Grade II - Slight weakness",
            2: "Grade III - Obvious weakness",
            3: "Grade IV - Disfiguring weakness",
            4: "Grade V - Barely perceptible",
            5: "Grade VI - No movement"
        }
    
    def _detect_landmarks(self, image):
        """Detect facial landmarks"""
        _, _, landmarks = self.mtcnn.detect(image, landmarks=True)
        
        if landmarks is None:
            h, w = image.shape[:2]
            scale_h, scale_w = h / 28, w / 28
            landmarks = np.array([[[9.528073 * scale_w, 11.062551 * scale_h],
                                  [21.396168 * scale_w, 10.919773 * scale_h],
                                  [15.380184 * scale_w, 17.380562 * scale_h],
                                  [10.255435 * scale_w, 22.121233 * scale_h],
                                  [20.583706 * scale_w, 22.25584 * scale_h]]])
        
        return landmarks[0].astype(int)
    
    def _extract_regions(self, image, landmarks, region_size=14):
        """Extract facial regions based on landmarks"""
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
        """Create hierarchical input for HTNet"""
        left_eye, right_eye, nose, left_mouth, right_mouth = regions
        
        left_eye_mouth = cv2.hconcat([left_eye, left_mouth])
        right_eye_mouth = cv2.hconcat([right_eye, right_mouth])
        combined = cv2.vconcat([left_eye_mouth, right_eye_mouth])
        
        return combined
    
    def preprocess(self, image_path):
        """Preprocess image for prediction"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        landmarks = self._detect_landmarks(image)
        regions = self._extract_regions(image, landmarks)
        hierarchical_input = self._create_hierarchical_input(regions)
        
        hierarchical_input = hierarchical_input.astype(np.float32) / 255.0
        hierarchical_input = torch.from_numpy(hierarchical_input).permute(2, 0, 1)
        hierarchical_input = hierarchical_input.unsqueeze(0)
        
        return hierarchical_input, image, landmarks
    
    def predict(self, image_path):
        """
        Predict facial palsy grade for an image
        
        Returns:
            grade (int): Predicted grade
            confidence (float): Confidence score
            probabilities (dict): Probabilities for all grades
        """
        input_tensor, original_image, landmarks = self.preprocess(image_path)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        grade = predicted.item()
        confidence = confidence.item()
        
        probs_dict = {i: probabilities[0][i].item() for i in range(self.num_classes)}
        
        return grade, confidence, probs_dict
    
    def visualize_prediction(self, image_path, output_path=None):
        """
        Visualize prediction with facial landmarks and grade
        """
        grade, confidence, probs = self.predict(image_path)
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size * 2, self.image_size * 2))
        
        grade_text = self.grade_names.get(grade, f"Grade {grade}")
        confidence_text = f"Confidence: {confidence:.2%}"
        
        cv2.putText(image, grade_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, confidence_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset = 90
        for i, prob in probs.items():
            text = f"Grade {i}: {prob:.2%}"
            cv2.putText(image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f'Visualization saved to {output_path}')
        else:
            cv2.imshow('Facial Palsy Prediction', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image


def batch_predict(predictor, image_dir, output_csv):
    """
    Predict for a batch of images and save results to CSV
    """
    import pandas as pd
    
    results = []
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f'Processing {len(image_files)} images...')
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            grade, confidence, probs = predictor.predict(img_path)
            results.append({
                'image': img_file,
                'predicted_grade': grade,
                'confidence': confidence,
                **{f'prob_grade_{i}': probs[i] for i in range(predictor.num_classes)}
            })
            print(f'{img_file}: Grade {grade} (confidence: {confidence:.2%})')
        except Exception as e:
            print(f'Error processing {img_file}: {e}')
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'\nResults saved to {output_csv}')


def main():
    parser = argparse.ArgumentParser(description='Facial Palsy Grading - Inference Demo')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to input image (for single prediction)')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory of images (for batch prediction)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                       help='CSV file to save batch predictions')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of facial palsy grades')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    predictor = FacialPalsyPredictor(
        model_path=args.model_path,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=args.device
    )
    
    if args.image_path:
        print(f'\nProcessing single image: {args.image_path}')
        grade, confidence, probs = predictor.predict(args.image_path)
        
        print(f'\nPrediction Results:')
        print(f'  Predicted Grade: {grade}')
        print(f'  Grade Name: {predictor.grade_names.get(grade, "Unknown")}')
        print(f'  Confidence: {confidence:.2%}')
        print(f'\nAll Probabilities:')
        for i, prob in probs.items():
            print(f'  Grade {i}: {prob:.2%}')
        
        if args.output_path or input('\nVisualize? (y/n): ').lower() == 'y':
            predictor.visualize_prediction(args.image_path, args.output_path)
    
    elif args.image_dir:
        print(f'\nProcessing batch of images from: {args.image_dir}')
        batch_predict(predictor, args.image_dir, args.output_csv)
    
    else:
        print('Please provide either --image_path or --image_dir')
        parser.print_help()


if __name__ == '__main__':
    main()
