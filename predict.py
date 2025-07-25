import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from skimage.feature import hog
import argparse
import os

class KidneyStonePredictor:
    def __init__(self, cnn_model_path='models/cnn_model.h5', 
                 svm_model_path='models/svm_model.pkl',
                 svm_scaler_path='models/svm_scaler.pkl'):
        self.cnn_model_path = cnn_model_path
        self.svm_model_path = svm_model_path
        self.svm_scaler_path = svm_scaler_path
        self.img_size = (224, 224)
        
        # Load models
        self.cnn_model = None
        self.svm_model = None
        self.svm_scaler = None
        
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            if os.path.exists(self.cnn_model_path):
                self.cnn_model = load_model(self.cnn_model_path)
                print("CNN model loaded successfully")
            else:
                print(f"CNN model not found at {self.cnn_model_path}")
            
            if os.path.exists(self.svm_model_path):
                self.svm_model = joblib.load(self.svm_model_path)
                print("SVM model loaded successfully")
            else:
                print(f"SVM model not found at {self.svm_model_path}")
            
            if os.path.exists(self.svm_scaler_path):
                self.svm_scaler = joblib.load(self.svm_scaler_path)
                print("SVM scaler loaded successfully")
            else:
                print(f"SVM scaler not found at {self.svm_scaler_path}")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            
            return img
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def extract_hog_features(self, image_path):
        """Extract HOG features for SVM prediction"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            img = cv2.resize(img, self.img_size)
            features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False)
            
            return features
        except Exception as e:
            print(f"Error extracting HOG features: {str(e)}")
            return None
    
    def predict_cnn(self, image_path):
        """Predict using CNN model"""
        if self.cnn_model is None:
            return None, "CNN model not loaded"
        
        img = self.preprocess_image(image_path)
        if img is None:
            return None, "Failed to preprocess image"
        
        try:
            img = np.expand_dims(img, axis=0)
            prediction = self.cnn_model.predict(img, verbose=0)[0][0]
            
            confidence = float(prediction)
            result = 'Stone' if prediction > 0.5 else 'Non-Stone'
            
            return {
                'prediction': result,
                'confidence': confidence,
                'model': 'CNN'
            }, None
            
        except Exception as e:
            return None, f"CNN prediction failed: {str(e)}"
    
    def predict_svm(self, image_path):
        """Predict using SVM model"""
        if self.svm_model is None or self.svm_scaler is None:
            return None, "SVM model or scaler not loaded"
        
        features = self.extract_hog_features(image_path)
        if features is None:
            return None, "Failed to extract HOG features"
        
        try:
            features = features.reshape(1, -1)
            features_scaled = self.svm_scaler.transform(features)
            
            prediction = self.svm_model.predict(features_scaled)[0]
            probabilities = self.svm_model.predict_proba(features_scaled)[0]
            
            confidence = float(max(probabilities))
            result = 'Stone' if prediction == 1 else 'Non-Stone'
            
            return {
                'prediction': result,
                'confidence': confidence,
                'model': 'SVM'
            }, None
            
        except Exception as e:
            return None, f"SVM prediction failed: {str(e)}"
    
    def predict_both(self, image_path):
        """Predict using both models"""
        results = {}
        
        # CNN prediction
        cnn_result, cnn_error = self.predict_cnn(image_path)
        if cnn_result:
            results['cnn'] = cnn_result
        else:
            results['cnn_error'] = cnn_error
        
        # SVM prediction
        svm_result, svm_error = self.predict_svm(image_path)
        if svm_result:
            results['svm'] = svm_result
        else:
            results['svm_error'] = svm_error
        
        return results
    
    def predict_batch(self, image_folder, model_type='both'):
        """Predict on a batch of images"""
        if not os.path.exists(image_folder):
            print(f"Folder {image_folder} does not exist")
            return []
        
        image_files = [f for f in os.listdir(image_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        results = []
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            print(f"Processing: {img_file}")
            
            if model_type == 'both':
                result = self.predict_both(img_path)
            elif model_type == 'cnn':
                result, error = self.predict_cnn(img_path)
                if error:
                    result = {'error': error}
            elif model_type == 'svm':
                result, error = self.predict_svm(img_path)
                if error:
                    result = {'error': error}
            else:
                result = {'error': 'Invalid model type'}
            
            results.append({
                'filename': img_file,
                'path': img_path,
                'results': result
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Kidney Stone Prediction')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--model', type=str, choices=['cnn', 'svm', 'both'], 
                       default='both', help='Model to use for prediction')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = KidneyStonePredictor()
    
    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"Image {args.image} does not exist")
            return
        
        print(f"Predicting on: {args.image}")
        
        if args.model == 'both':
            results = predictor.predict_both(args.image)
            print("\nPrediction Results:")
            print("-" * 40)
            
            for model_name, result in results.items():
                if 'error' not in model_name:
                    print(f"{result['model']}:")
                    print(f"  Prediction: {result['prediction']}")
                    print(f"  Confidence: {result['confidence']:.4f}")
                else:
                    print(f"Error: {result}")
        
        elif args.model == 'cnn':
            result, error = predictor.predict_cnn(args.image)
            if result:
                print(f"\nCNN Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
            else:
                print(f"Error: {error}")
        
        elif args.model == 'svm':
            result, error = predictor.predict_svm(args.image)
            if result:
                print(f"\nSVM Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
            else:
                print(f"Error: {error}")
    
    elif args.folder:
        # Batch prediction
        print(f"Batch prediction on folder: {args.folder}")
        results = predictor.predict_batch(args.folder, args.model)
        
        print(f"\nProcessed {len(results)} images:")
        print("-" * 60)
        
        for result in results:
            print(f"\nFile: {result['filename']}")
            
            if 'error' in result['results']:
                print(f"  Error: {result['results']['error']}")
            else:
                for model_name, pred_result in result['results'].items():
                    if 'error' not in model_name:
                        print(f"  {pred_result['model']}: {pred_result['prediction']} "
                              f"(confidence: {pred_result['confidence']:.4f})")
    
    else:
        print("Please provide either --image or --folder argument")

if __name__ == "__main__":
    main()
