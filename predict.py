"""
Lung Cancer Prediction - Inference Script
This script uses the pre-trained model to predict lung cancer types from CT scan images.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import from different Keras versions
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    print("Using TensorFlow Keras")
except ImportError:
    try:
        from keras.models import load_model
        from keras.preprocessing import image
        print("Using Keras 3")
    except ImportError:
        print("Error: Neither TensorFlow nor Keras is installed!")
        print("Please install one of them:")
        print("  pip install tensorflow")
        print("  or")
        print("  pip install keras")
        sys.exit(1)

# Define constants
IMAGE_SIZE = (350, 350)
MODEL_PATH = 'best_model.hdf5'

# Class labels (in alphabetical order as they appear in the dataset)
CLASS_LABELS = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

# Simplified class names for display
SIMPLIFIED_LABELS = {
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'Adenocarcinoma',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'Large Cell Carcinoma',
    'normal': 'Normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'Squamous Cell Carcinoma'
}


def load_and_preprocess_image(img_path, target_size):
    """
    Load and preprocess an image for prediction.
    
    Args:
        img_path: Path to the image file
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array ready for prediction
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array


def load_model_with_weights(model_path, image_size=(350, 350)):
    """
    Load the model by rebuilding the architecture and loading weights.
    The HDF5 file contains only weights, not the full model.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
    print(f"Rebuilding model architecture...")
    
    # Define the model architecture (same as training)
    OUTPUT_SIZE = 4
    
    # Load pre-trained Xception model
    pretrained_model = tf.keras.applications.Xception(
        weights='imagenet', 
        include_top=False, 
        input_shape=[*image_size, 3]
    )
    pretrained_model.trainable = False
    
    # Build the model
    model = Sequential([
        pretrained_model,
        GlobalAveragePooling2D(),
        Dense(OUTPUT_SIZE, activation='softmax')
    ])
    
    # Do NOT compile the model for inference (saves memory)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Loading weights from '{model_path}'...")
    
    # Load weights
    try:
        model.load_weights(model_path)
        print("✓ Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise
    
    return model


def predict_image(model, img_path, show_plot=True):
    """
    Predict the class of a lung cancer CT scan image.
    
    Args:
        model: Loaded Keras model
        img_path: Path to the image file
        show_plot: Whether to display the image with prediction
    
    Returns:
        Dictionary containing prediction results
    """
    # Load and preprocess the image
    img = load_and_preprocess_image(img_path, IMAGE_SIZE)
    
    # Make prediction
    predictions = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get class label
    predicted_label = CLASS_LABELS[predicted_class_idx]
    simplified_label = SIMPLIFIED_LABELS[predicted_label]
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted Class: {simplified_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"{'='*60}")
    
    # Show all class probabilities
    print("\nAll Class Probabilities:")
    for i, label in enumerate(CLASS_LABELS):
        prob = predictions[0][i] * 100
        simplified = SIMPLIFIED_LABELS[label]
        print(f"  {simplified:30s}: {prob:6.2f}%")
    
    # Display the image with prediction
    if show_plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
        plt.title(f"Predicted: {simplified_label}\nConfidence: {confidence:.2f}%", 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return {
        'predicted_class': simplified_label,
        'confidence': confidence,
        'all_probabilities': {SIMPLIFIED_LABELS[label]: predictions[0][i] * 100 
                             for i, label in enumerate(CLASS_LABELS)}
    }


def predict_batch(model, image_paths, show_plots=True):
    """
    Predict multiple images at once.
    
    Args:
        model: Loaded Keras model
        image_paths: List of image file paths
        show_plots: Whether to display images with predictions
    
    Returns:
        List of prediction results
    """
    results = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            result = predict_image(model, img_path, show_plot=show_plots)
            results.append(result)
        else:
            print(f"\nWarning: Image not found - {img_path}")
    
    return results


def main():
    """Main function to run the inference script."""
    
    print("\n" + "="*60)
    print("  Lung Cancer Prediction - Inference Script")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model file '{MODEL_PATH}' not found!")
        print("Please ensure the model file is in the current directory.")
        sys.exit(1)
    
    # Load the pre-trained model
    print(f"\nLoading model from '{MODEL_PATH}'...")
    try:
        model = load_model_with_weights(MODEL_PATH, IMAGE_SIZE)
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Predict images provided as command line arguments
        image_paths = sys.argv[1:]
        print(f"\nPredicting {len(image_paths)} image(s)...")
        predict_batch(model, image_paths, show_plots=True)
    else:
        # Interactive mode
        print("\n" + "-"*60)
        print("Usage Options:")
        print("-"*60)
        print("1. Command line: python predict.py <image_path1> <image_path2> ...")
        print("2. Interactive mode: Enter image paths when prompted")
        print("-"*60)
        
        while True:
            print("\n" + "="*60)
            img_path = input("\nEnter image path (or 'quit' to exit): ").strip()
            
            if img_path.lower() in ['quit', 'exit', 'q']:
                print("\nExiting... Goodbye!")
                break
            
            if not img_path:
                print("Please enter a valid image path.")
                continue
            
            # Remove quotes if present
            img_path = img_path.strip('"').strip("'")
            
            if os.path.exists(img_path):
                predict_image(model, img_path, show_plot=True)
            else:
                print(f"\nError: Image file not found - {img_path}")
                print("Please check the path and try again.")


if __name__ == "__main__":
    main()
