import torch
import numpy as np
import random
import pandas as pd
import os
from pathlib import Path
from ultralytics import YOLO

def set_seed(seed=42):
    """Set random seeds for reproducible initialization"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (optional, may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_yolo_and_save_excel(model, test_folder, excel_filename="yolo_test_results.xlsx"):
    """Test YOLO model on test folder and save results to Excel"""
    
    if not os.path.exists(test_folder):
        print(f"Test folder does not exist: {test_folder}")
        return []
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    # Get class names from the model
    class_names = model.names
    
    print(f"Testing model on: {test_folder}")
    print(f"Available classes: {class_names}")
    
    # Iterate through class folders
    for class_name in os.listdir(test_folder):
        class_folder = os.path.join(test_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
            
        print(f"\nProcessing class: {class_name}")
        
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            
            # Skip non-image files
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue
                
            try:
                # Run prediction on single image
                prediction_results = model.predict(img_path, verbose=False)
                
                if prediction_results and len(prediction_results) > 0:
                    result = prediction_results[0]
                    
                    # Get prediction details
                    if hasattr(result, 'probs') and result.probs is not None:
                        # Classification results
                        probs = result.probs.data.cpu().numpy()
                        predicted_class_idx = np.argmax(probs)
                        predicted_class = class_names[predicted_class_idx]
                        probability = float(probs[predicted_class_idx])
                        
                        # Check if prediction is correct
                        is_correct = predicted_class == class_name
                        if is_correct:
                            correct_predictions += 1
                        total_predictions += 1
                        
                        # Store result
                        result_dict = {
                            "image_name": img_name,
                            "image_path": img_path,
                            "true_class": class_name,
                            "predicted_class": predicted_class,
                            "probability": probability,
                            "correct": is_correct
                        }
                        results.append(result_dict)
                        
                        # Print result for each image
                        status = "[OK]" if is_correct else "[X]"
                        print(f"{status} {img_name}: True={class_name}, Predicted={predicted_class}, Prob={probability:.3f}")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Calculate and display overall accuracy
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n" + "="*60)
        print(f"TESTING RESULTS:")
        print(f"Total images tested: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Incorrect predictions: {total_predictions - correct_predictions}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"="*60)
    
    # Save results to Excel
    save_results_to_excel(results, excel_filename)
    
    # Display per-class accuracy
    display_confusion_info(results)
    
    return results, accuracy if total_predictions > 0 else 0

def save_results_to_excel(results, filename="yolo_test_results.xlsx"):
    """Save test results to Excel file"""
    if not results:
        print("No results to save!")
        return
    
    # Create DataFrame with required columns
    df_data = []
    for result in results:
        df_data.append({
            "image_name": result["image_name"],
            "true_class": result["true_class"],
            "predicted_class": result["predicted_class"],
            "probability": round(result["probability"], 4)
        })
    
    df = pd.DataFrame(df_data)
    
    # Save to Excel
    df.to_excel(filename, index=False)
    print(f"\nResults saved to: {filename}")
    print(f"Total records saved: {len(df)}")

def display_confusion_info(results):
    """Display per-class accuracy and confusion information"""
    from collections import defaultdict
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for result in results:
        true_class = result["true_class"]
        class_total[true_class] += 1
        if result["correct"]:
            class_correct[true_class] += 1
    
    print(f"\nPER-CLASS ACCURACY:")
    print(f"{'-'*50}")
    for class_name in sorted(class_total.keys()):
        accuracy = (class_correct[class_name] / class_total[class_name]) * 100
        print(f"{class_name}: {class_correct[class_name]}/{class_total[class_name]} ({accuracy:.1f}%)")

# Set seed before any model operations
set_seed(42)

# Load trained model
model = YOLO("runs/classify/test_architecture/weights/best.pt")

# Set seed before validation for consistent results
set_seed(42)

# Run standard validation on test set
print("Running standard YOLO validation...")
metrics = model.val(split='test')
print(f"Standard validation Top1 Accuracy: {metrics.top1 if hasattr(metrics, 'top1') else 'N/A'}")
print(f"Standard validation Top5 Accuracy: {metrics.top5 if hasattr(metrics, 'top5') else 'N/A'}")

# Run detailed testing and save to Excel
print("\n" + "="*60)
print("Running detailed testing for Excel export...")
print("="*60)

# Define test folder path (adjust according to your dataset structure)
test_folder = "path/to/your/dataset/test"

# Run detailed testing and save results
set_seed(42)
results_filename = "aggrnet_test_results.xlsx"
detailed_results, detailed_accuracy = test_yolo_and_save_excel(
    model, 
    test_folder, 
    excel_filename=results_filename
)

print(f"\nDetailed testing completed!")
print(f"Results saved to: {results_filename}")
