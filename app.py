import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from predict import load_model_with_weights, predict_image, MODEL_PATH, IMAGE_SIZE

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = None
try:
    print("Loading model...")
    # Ensure the model path is correct relative to this script
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file {MODEL_PATH} not found.")
    else:
        model = load_model_with_weights(MODEL_PATH, IMAGE_SIZE)
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded correctly. Please check server logs.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict
            result = predict_image(model, filepath, show_plot=False)
            
            # Clean up - optional, maybe we want to keep them? For now, let's keep them or delete?
            # User didn't specify, but usually good to clean up in a simple demo app or use a temp dir.
            # I'll keep it for now as it might be useful for debugging, but maybe delete to save space?
            # Let's delete to be clean.
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'prediction': result['predicted_class'],
                'confidence': f"{result['confidence']:.2f}%",
                'probabilities': result['all_probabilities']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
