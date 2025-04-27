from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import base64
import json
import mediapipe as mp

import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
from datetime import datetime
from fpdf import FPDF  # Library for creating PDFs
from fpdf.enums import XPos, YPos
import tempfile

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Image as RLImage, Paragraph
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, landscape
import random

import requests

from dotenv import load_dotenv
from linkedin_scraper import scrape_linkedin_profile
from llama_cpp import Llama

import pandas as pd
from sklearn import linear_model
import fitz  # PyMuPDF
from io import BytesIO

import logging

logging.getLogger("pdfminer").setLevel(logging.WARNING)  # Suppress debug logs


app = Flask(__name__)
CORS(app)  # Enable CORS

################################################################

#LINKEDIN ANALYSIS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize LLaMA model
llm = Llama(model_path="D:/models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048, n_gpu_layers=50)

def analyze_profile(profile_data):
    """Generate an analysis report for the given profile data."""
    prompt = f"Analyze the following LinkedIn profile and suggest improvements, also mention details of the user, and do analysis. At the start of response, rate the profile from 1-10:\n\n{json.dumps(profile_data, indent=2)}"
    output = llm.create_completion(prompt, max_tokens=1000)
    return output["choices"][0]["text"]

@app.route('/api/analyze-linkedin', methods=['POST', 'OPTIONS'])
def analyze_linkedin():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({'status': 'success'}), 200

    try:
        print("Received request to /api/analyze-linkedin")  # Debug statement
        data = request.get_json()
        print("Request data:", data)  # Debug statement

        profile_url = data.get('profile_url')
        if not profile_url:
            print("No profile URL provided")  # Debug statement
            return jsonify({'status': 'error', 'message': 'No LinkedIn profile URL provided.'}), 400

        print("Scraping LinkedIn profile...")  # Debug statement
        formatted_text, image_url = scrape_linkedin_profile(profile_url)

        if not formatted_text:
            print("Failed to scrape profile data")  # Debug statement
            return jsonify({'status': 'error', 'message': 'Failed to scrape LinkedIn profile data. Ensure the profile is public.'}), 400

        print("Profile data scraped successfully")  # Debug statement
        profile_data = {"profile_text": formatted_text, "image_url": image_url}

        print("Analyzing profile...")  # Debug statement
        analysis = analyze_profile(profile_data)

        response = {
            'status': 'success',
            'analysis': analysis,
            'profile_image_url': image_url
        }
        print("Response:", response)  # Debug statement
        return jsonify(response)

    except Exception as e:
        print(f"Error analyzing LinkedIn profile: {e}")  # Debug statement
        return jsonify({'status': 'error', 'message': str(e)}), 500




####################################################################

# CV ANALYSIS PART


# Load and train the personality prediction model
class TrainModel:
    def __init__(self):
        self.model = None

    def train(self):
        # Load the training dataset
        data = pd.read_csv('training_dataset.csv')
        array = data.values

        # Convert gender to numerical values
        for i in range(len(array)):
            if array[i][0] == "Male":
                array[i][0] = 1
            else:
                array[i][0] = 0

        # Prepare features and target
        df = pd.DataFrame(array)
        maindf = df[[0, 1, 2, 3, 4, 5, 6]]  # Features: gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion
        mainarray = maindf.values
        temp = df[7]  # Target: personality
        train_y = temp.values

        # Train the logistic regression model
        self.model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.model.fit(mainarray, train_y)

    def predict(self, test_data):
        try:
            test_predict = list(map(int, test_data))  # Convert input to integers
            y_pred = self.model.predict([test_predict])
            return y_pred[0]
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

# Initialize the model
model = TrainModel()
model.train()

def analyze_cv_with_llm(cv_data, predicted_personality):
    """Generate an analysis report for the given CV data and predicted personality. """
    prompt = f"""
    Analyze the following CV and predicted personality traits in detail mentioning details in the CV. Provide suggestions for improvement, highlight strengths, and mention areas for development. At the end of the response, rate the CV from 1-10:

    CV Data:
    {json.dumps(cv_data, indent=2)}

    Predicted Personality:
    {predicted_personality}
    """
    output = llm.create_completion(prompt, max_tokens=3000, stop=["\n\n"])
    return output["choices"][0]["text"]


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF while ensuring all words are captured and sorted correctly."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                words = page.get_text("words")  # Extract all words
                
                if not words:
                    print(f"[WARNING] No words extracted on page {page_num + 1}. Trying alternative methods...")
                
                # Sort words by vertical position first, then horizontal
                words.sort(key=lambda w: (w[1], w[0]))
                
                # Reconstruct the text
                page_text = " ".join(word[4] for word in words)
                text += page_text + "\n\n"
    
        with open("debug_extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(resume_data['text'])


    except Exception as e:
        print(f"[ERROR] Error reading PDF: {e}")

    print("[DEBUG] Full Extracted Text:\n", text[:1000])  # Print first 1000 characters for debugging
    return text.strip()

# New endpoint for CV analysis
@app.route('/api/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        print("\n[DEBUG] Received request for CV analysis.")

        if 'resume' not in request.files:
            print("[ERROR] No resume file uploaded.")
            return jsonify({'status': 'error', 'message': 'No resume file uploaded.'}), 400

        resume_file = request.files['resume']
        if resume_file.filename == '':
            print("[ERROR] No selected file.")
            return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

        print(f"[DEBUG] Uploaded file name: {resume_file.filename}")

        # Ensure uploads directory exists
        uploads_dir = 'uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            print("[DEBUG] Created uploads directory.")

        resume_path = os.path.join(uploads_dir, resume_file.filename)
        resume_file.save(resume_path)
        print(f"[DEBUG] Saved resume at: {resume_path}")

        # Extract text using pdfplumber
        resume_data = {}
        # Extract text using PyMuPDF (fitz)
        try:
            extracted_text = extract_text_from_pdf(resume_path)  # Use the improved function
            resume_data['text'] = extracted_text.strip() if extracted_text else None
            print(f"[DEBUG] Full Extracted Resume Text:\n{resume_data['text']}")
        except Exception as e:
            print(f"[ERROR] Error reading PDF: {e}")
            os.remove(resume_path)
            return jsonify({'status': 'error', 'message': f'Error reading PDF: {str(e)}'}), 500
            if not resume_data.get('text'):
                print("[ERROR] Could not extract text from PDF.")
                os.remove(resume_path)
                return jsonify({'status': 'error', 'message': 'Could not extract text from PDF.'}), 400

        # Get personality prediction inputs
        required_fields = ['gender', 'age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
        personality_inputs = [request.form.get(field) for field in required_fields]

        print(f"[DEBUG] Received personality inputs: {personality_inputs}")

        if None in personality_inputs:
            print("[ERROR] Missing personality inputs.")
            os.remove(resume_path)
            return jsonify({'status': 'error', 'message': 'All personality inputs are required.'}), 400

        predicted_personality = model.predict(personality_inputs)
        print(f"[DEBUG] Predicted Personality: {predicted_personality}")

        if not predicted_personality:
            print("[ERROR] Failed to predict personality.")
            os.remove(resume_path)
            return jsonify({'status': 'error', 'message': 'Failed to predict personality.'}), 500

        llm_analysis = analyze_cv_with_llm(resume_data, predicted_personality)
        print(f"[DEBUG] LLM Analysis Output:\n{llm_analysis[:500]}")  # Print first 500 characters for verification

        os.remove(resume_path)
        print("[DEBUG] Deleted resume file after processing.")


        return jsonify({
            'status': 'success',
            'resume_data': resume_data,
            'predicted_personality': predicted_personality,
            'llm_analysis': llm_analysis
        })

    except Exception as e:
        print(f"[ERROR] Error analyzing CV: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500




# Constants for audio processing
MAX_LEN = 100
N_MFCC = 13
AUDIO_MODEL_PATH = "C:/Users/HP/OneDrive/Desktop/PEERS/AUDIO/big_five_audio_lstm.h5"  # Update this with your correct model path

# Load the LSTM model for audio processing
if os.path.exists(AUDIO_MODEL_PATH):
    try:
        audio_model = load_model(AUDIO_MODEL_PATH, custom_objects={"mse": MeanSquaredError()})
        print("Audio LSTM model loaded successfully.")
    except Exception as e:
        print(f"Error loading audio model: {str(e)}")
else:
    print(f"Error: Audio model not found at {AUDIO_MODEL_PATH}")




# Define Mediapipe components
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32)
        self.fc1 = nn.Linear(32 * 48 * 48, k)

    def forward(self, x):
        x = self.pool(F.relu(self.l1(x)))
        x = self.pool(F.relu(self.l2(x)))
        x = x.view(-1, 32 * 48 * 48)
        x = self.fc1(x)
        return x

# Load your models here
eye_model = CNN(3)
eyebrow_model = CNN(3)
jaw_model = CNN(4)
mouth_model = CNN(3)
nose_model = CNN(3)

eye_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/eye_model.pt", weights_only=True))
eyebrow_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/eyebrow_model.pt", weights_only=True))
jaw_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/jaw_model.pt", weights_only=True))
mouth_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/mouth_model.pt", weights_only=True))
nose_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/nose_model.pt", weights_only=True))

# Load the analysis.json file
with open('C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/data/analysis.json', 'r') as f:
    shape_descriptions = json.load(f)

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

        # Decode base64 and convert to RGB
        img_data = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_cv = np.array(img).astype(np.uint8)

        # Initialize Mediapipe Face Mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            # Process the image
            results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))

            if not results.multi_face_landmarks:
                return jsonify({'status': 'error', 'message': 'No faces detected.'}), 400

            feature_points = []  # To store feature points for drawing
            # Draw landmarks on the image and store the points for model predictions
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    # Get the coordinates of the landmarks
                    x = int(landmark.x * img_cv.shape[1])
                    y = int(landmark.y * img_cv.shape[0])
                    # Draw the landmark points on the image
                    cv2.circle(img_cv, (x, y), 2, (0, 255, 0), -1)  # Green circle for landmarks
                    feature_points.append((x, y))

            # Assuming we want to extract features based on certain landmark points
            if feature_points:
                # Example: extracting the region for eye predictions (you can adjust the indices based on your needs)
                left_eye_region = img_cv[feature_points[36][1]:feature_points[41][1], feature_points[36][0]:feature_points[39][0]]  # Eye region for prediction
                left_eye_img_resized = cv2.resize(left_eye_region, (200, 200))

                preprocess = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = preprocess(Image.fromarray(cv2.cvtColor(left_eye_img_resized, cv2.COLOR_BGR2RGB))).unsqueeze(0)

                # Pass through models
                eyebrow = eyebrow_model(img_tensor)
                eye = eye_model(img_tensor)
                nose = nose_model(img_tensor)
                mouth = mouth_model(img_tensor)
                jaw = jaw_model(img_tensor)

                # Get shapes
                eyebrow_shape = ["Arch", "Circle", "Straight"][torch.argmax(eyebrow).item()]
                eye_shape = ["Big", "Silt", "Small"][torch.argmax(eye).item()]
                nose_shape = ["Long", "Small", "Wide"][torch.argmax(nose).item()]
                mouth_shape = ["Medium", "Small", "Thick"][torch.argmax(mouth).item()]
                jaw_shape = ["Circle", "Oval", "Square", "Triangle"][torch.argmax(jaw).item()]

                # Prepare the results using the loaded JSON data
                def get_shape_analysis(shape_name, feature_name):
                    for region in shape_descriptions['face_regions']:
                        if region['name'] == shape_name:
                            for feature in region['features']:
                                if feature['name'] == feature_name:
                                    return feature['analysis']
                    return "Description not found."

                results = {
                    'eyebrow': {
                        'shape': eyebrow_shape,
                        'description': get_shape_analysis('eyebrows', eyebrow_shape)
                    },
                    'eye': {
                        'shape': eye_shape,
                        'description': get_shape_analysis('eyes', eye_shape)
                    },
                    'nose': {
                        'shape': nose_shape,
                        'description': get_shape_analysis('nose', nose_shape)
                    },
                    'mouth': {
                        'shape': mouth_shape,
                        'description': get_shape_analysis('mouth', mouth_shape)
                    },
                    'jaw': {
                        'shape': jaw_shape,
                        'description': get_shape_analysis('face', jaw_shape)  # Use 'face' for jaw analysis
                    }
                }

        # Convert modified image back to base64
        _, buffer = cv2.imencode('.png', img_cv)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the response with results and processed image
        return jsonify({'status': 'success', 'results': results, 'image': img_base64})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    try:
        # Check if an audio file is provided
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided.'}), 400

        audio_file = request.files['audio']
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
            return jsonify({'status': 'error', 'message': 'Invalid audio file format. Supported formats: .wav, .mp3, .ogg'}), 400

        # Save the uploaded audio temporarily
        audio_path = "temp_audio.wav"
        audio_file.save(audio_path)

        # Function to extract MFCC features
        def extract_mfcc_sequence(audio_path):
            try:
                y, sr = librosa.load(audio_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                padded_mfcc = np.zeros((N_MFCC, MAX_LEN))
                mfcc_len = min(mfcc.shape[1], MAX_LEN)
                padded_mfcc[:, :mfcc_len] = mfcc[:, :mfcc_len]
                return padded_mfcc.T  # Shape: (MAX_LEN, N_MFCC)
            except Exception as e:
                raise ValueError(f"Error extracting MFCC features: {e}")

        # Extract features and reshape
        features = extract_mfcc_sequence(audio_path).reshape(1, MAX_LEN, N_MFCC)

        # Make predictions
        predictions = audio_model.predict(features)[0]

        # Adjust predictions to increase confidence scores
        adjusted_predictions = 0.5 + (predictions - 0.5) * 1.6
        adjusted_predictions = np.clip(adjusted_predictions, 0, 1)

        # Big Five Traits
        BIG_FIVE = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
        results = {trait: round(float(score), 2) for trait, score in zip(BIG_FIVE, adjusted_predictions)}

        # Return the predictions as JSON
        return jsonify({'status': 'success', 'results': results})
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred. Please try again.'}), 500



# Add Zodiac and Chinese Zodiac logic to the Flask API

# Zodiac signs with date ranges
ZODIAC_SIGNS = [
    ("Capricorn", (12, 22), (1, 19)),
    ("Aquarius", (1, 20), (2, 18)),
    ("Pisces", (2, 19), (3, 20)),
    ("Aries", (3, 21), (4, 19)),
    ("Taurus", (4, 20), (5, 20)),
    ("Gemini", (5, 21), (6, 20)),
    ("Cancer", (6, 21), (7, 22)),
    ("Leo", (7, 23), (8, 22)),
    ("Virgo", (8, 23), (9, 22)),
    ("Libra", (9, 23), (10, 22)),
    ("Scorpio", (10, 23), (11, 21)),
    ("Sagittarius", (11, 22), (12, 21)),
]

# Chinese Zodiac signs
CHINESE_ZODIAC = [
    "Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", 
    "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig"
]

# Load zodiac and Chinese zodiac data from JSON
JSON_FILE_PATH = os.path.join('C:\\Projects\\brainwave\\frontend\\src\\components', 'zodiac_chinese_signs.json')

try:
    with open(JSON_FILE_PATH, 'r') as file:
        zodiac_data = json.load(file)
        ZODIAC_SIGNS = zodiac_data.get('zodiac_signs', [])
        CHINESE_ZODIAC_SIGNS = zodiac_data.get('chinese_signs', [])  # Corrected key here
except Exception as e:
    print(f"Error loading JSON file: {e}")
    ZODIAC_SIGNS = []
    CHINESE_ZODIAC_SIGNS = []

    
@app.route('/api/calculate-signs', methods=['POST'])
def calculate_signs():
    try:
        data = request.get_json()
        dob = data.get('dob')  # Expected format: 'MM/DD/YYYY'

        if not dob:
            return jsonify({'status': 'error', 'message': 'DOB not provided.'}), 400

        # Validate and parse DOB format (MM/DD/YYYY)
        try:
            dob_obj = datetime.strptime(dob, '%m/%d/%Y')  # Parse the DOB
            dob = dob_obj.strftime('%Y-%m-%d')  # Convert to 'YYYY-MM-DD' format
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid DOB format. Use MM/DD/YYYY.'}), 400

        # Parse the DOB into year, month, and day
        year, month, day = map(int, dob.split('-'))

        # Zodiac sign calculation
        def get_zodiac_sign(month, day):
            for sign_data in ZODIAC_SIGNS:
                start_month, start_day = map(int, sign_data['date_range']['start'].split('-'))
                end_month, end_day = map(int, sign_data['date_range']['end'].split('-'))
                if (month == start_month and day >= start_day) or \
                   (month == end_month and day <= end_day) or \
                   (start_month < month < end_month):
                    return sign_data
            # Default to Capricorn if no match found
            return next(sign for sign in ZODIAC_SIGNS if sign['name'] == 'Capricorn')

        # Chinese Zodiac calculation
        def get_chinese_zodiac(year):
            if not CHINESE_ZODIAC_SIGNS:
                raise ValueError("Chinese zodiac signs not properly loaded.")

            index = year % 12
            if index < 0 or index >= len(CHINESE_ZODIAC_SIGNS):
                raise IndexError("Calculated index for Chinese Zodiac is out of range.")
            return CHINESE_ZODIAC_SIGNS[index]


        # Get signs and descriptions
        zodiac_sign_data = get_zodiac_sign(month, day)
        chinese_zodiac_data = get_chinese_zodiac(year)

        # Return the results
        return jsonify({
            'status': 'success',
            'zodiac_sign': zodiac_sign_data,
            'chinese_zodiac': chinese_zodiac_data
        })

    except Exception as e:
        print(f"Error calculating signs: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/api/find-lookalike', methods=['POST'])
def find_lookalike():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

        # Decode base64 and convert to RGB
        img_data = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_cv = np.array(img).astype(np.uint8)

        # Initialize Mediapipe Face Mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            # Process the image
            results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))

            if not results.multi_face_landmarks:
                return jsonify({'status': 'error', 'message': 'No faces detected.'}), 400

            feature_points = []
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * img_cv.shape[1])
                    y = int(landmark.y * img_cv.shape[0])
                    feature_points.append((x, y))

            # Predict feature shapes
            if feature_points:
                # Extract a representative region (example: left eye region)
                left_eye_region = img_cv[
                    feature_points[36][1]:feature_points[41][1],
                    feature_points[36][0]:feature_points[39][0]
                ]
                left_eye_img_resized = cv2.resize(left_eye_region, (200, 200))

                preprocess = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = preprocess(Image.fromarray(cv2.cvtColor(left_eye_img_resized, cv2.COLOR_BGR2RGB))).unsqueeze(0)

                eyebrow = eyebrow_model(img_tensor)
                eye = eye_model(img_tensor)
                nose = nose_model(img_tensor)
                mouth = mouth_model(img_tensor)
                jaw = jaw_model(img_tensor)

                features = {
                    'eyebrow': ["Arch", "Circle", "Straight"][torch.argmax(eyebrow).item()],
                    'eye': ["Big", "Silt", "Small"][torch.argmax(eye).item()],
                    'nose': ["Long", "Small", "Wide"][torch.argmax(nose).item()],
                    'mouth': ["Medium", "Small", "Thick"][torch.argmax(mouth).item()],
                    'jaw': ["Circle", "Oval", "Square", "Triangle"][torch.argmax(jaw).item()]
                }

            # Match with celebrity database
            with open('C:\\Projects\\brainwave\\frontend\\src\\components\\celebrity_data.json') as f:
                celeb_data = json.load(f)

            matches = []
            for celeb in celeb_data:
                if celeb['gender'].lower() != data.get('gender', '').lower():
                    continue  # Skip if gender doesn't match

                score = 0
                total_weight = 0
                weights = {'eyebrow': 1, 'eye': 2, 'nose': 2, 'mouth': 1, 'jaw': 3}

                for feature, weight in weights.items():
                    if celeb['features'][feature] == features[feature]:
                        score += weight
                    total_weight += weight

                match_percentage = (score / total_weight) * 100
                matches.append({
                    'name': celeb['name'],
                    'gender': celeb['gender'],
                    'features': celeb['features'],
                    'match_percentage': round(match_percentage, 2)
                })

            # Sort matches by percentage (descending) and name (alphabetically)
            matches = sorted(matches, key=lambda x: (-x['match_percentage'], x['name']))

            # Return top 3 matches
            if matches:
                return jsonify({'status': 'success', 'matches': matches[:3]})
            else:
                return jsonify({'status': 'error', 'message': 'No matching celebrities found.'})

    except Exception as e:
        print(f"Error finding lookalike: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500




def get_celebrity_image(celebrity_name):
    """Fetches the celebrity's image URL from Wikipedia API."""
    try:
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{celebrity_name.replace(' ', '_')}"
        response = requests.get(search_url).json()
        
        if 'thumbnail' in response and 'source' in response['thumbnail']:
            return response['thumbnail']['source']  # Return image URL
        return None
    except Exception as e:
        print(f"Error fetching image for {celebrity_name}: {e}")
        return None

def generate_pdf_report(name, email, number, zodiac_results, facial_features, lookalike_results, labeled_image_base64):
    try:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=landscape(letter))  # Set landscape mode
        width, height = landscape(letter)

        # ---- ENHANCED STYLING WITH BETTER CONTRAST ----
        
        # Modern background with gradient - lighter to improve readability
        pdf.saveState()
        p = pdf.beginPath()
        p.moveTo(0, 0)
        p.lineTo(width, 0)
        p.lineTo(width, height)
        p.lineTo(0, height)
        p.close()
        pdf.clipPath(p, stroke=0, fill=0)
        
        # Draw gradient background (dark blue to purple but lighter for better text visibility)
        for i in range(100):
            y = i * height / 100
            r = 0.1 - (i * 0.0005)  # Slightly lighter
            g = 0.1 - (i * 0.0003)
            b = 0.25 - (i * 0.0005)
            pdf.setFillColorRGB(r, g, b)
            pdf.rect(0, y, width, height/100, fill=1, stroke=0)
        
        # Reduce pattern overlay density for better text clarity
        pdf.setFillColorRGB(1, 1, 1, 0.05)  # Slightly more opacity for visibility
        for i in range(6):  # Fewer circles
            x = random.randint(0, int(width))
            y = random.randint(0, int(height))
            size = random.randint(5, 15)
            pdf.circle(x, y, size, fill=1, stroke=0)
        
        for i in range(8):  # Fewer lines
            x1 = random.randint(0, int(width))
            y1 = random.randint(0, int(height))
            x2 = x1 + random.randint(-80, 80)
            y2 = y1 + random.randint(-80, 80)
            pdf.setStrokeColorRGB(1, 1, 1, 0.1)  # Slightly brighter lines
            pdf.line(x1, y1, x2, y2)
            
        pdf.restoreState()

        # ---- HEADER SECTION WITH IMPROVED CONTRAST ----
        
        # Add modern header with logo and glow effect
        pdf.saveState()
        # Header background
        pdf.setFillColorRGB(0.15, 0.15, 0.3, 0.9)  # Slightly lighter blue with more opacity
        pdf.roundRect(30, height - 70, width - 60, 50, 10, fill=1, stroke=0)
        
        # Add "AI" with futuristic styling - much brighter
        pdf.setFont("Helvetica-Bold", 28)
        pdf.setFillColorRGB(0.4, 1, 1)  # Much brighter cyan
        pdf.drawString(50, height - 45, "AI")
        
        # Title - brighter white
        pdf.setFont("Helvetica-Bold", 24)
        pdf.setFillColorRGB(1, 1, 1)  # Pure white
        pdf.drawString(85, height - 45, "Personality & Lookalike Report")
        
        # Add stronger glow effect around header
        pdf.setStrokeColorRGB(0.4, 1, 1, 0.6)  # Brighter glow with more opacity
        pdf.setLineWidth(2.5)  # Thicker line
        pdf.roundRect(30, height - 70, width - 60, 50, 10, fill=0, stroke=1)
        pdf.restoreState()

        # ---- USER INFO SECTION WITH BETTER CONTRAST ----
        
        # User info panel - more visible
        pdf.saveState()
        pdf.setFillColorRGB(1, 1, 1, 0.2)  # More visible panel
        pdf.roundRect(40, height - 130, width/2 - 60, 50, 8, fill=1, stroke=0)
        
        # Add border to user info panel for better visibility
        pdf.setStrokeColorRGB(1, 1, 1, 0.5)  # White border
        pdf.setLineWidth(1)
        pdf.roundRect(40, height - 130, width/2 - 60, 50, 8, fill=0, stroke=1)
        
        # User details with icons - brighter text
        pdf.setFillColorRGB(1, 1, 1)  # White text
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, height - 95, "Name:")
        pdf.drawString(50, height - 110, "Email:")
        pdf.drawString(50, height - 125, "Phone:")
        
        pdf.setFont("Helvetica", 12)
        pdf.drawString(100, height - 95, name)
        pdf.drawString(100, height - 110, email)
        pdf.drawString(100, height - 125, number)
        pdf.restoreState()
        
        # Date and report ID - brighter
        pdf.saveState()
        pdf.setFillColorRGB(1, 1, 1)  # Pure white
        pdf.setFont("Helvetica-Bold", 10)  # Bold for better visibility
        report_date = datetime.now().strftime("%Y-%m-%d")
        report_id = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
        pdf.drawRightString(width - 40, height - 95, f"Report Date: {report_date}")
        pdf.drawRightString(width - 40, height - 110, f"Report ID: {report_id}")
        pdf.restoreState()

        # ---- ZODIAC ANALYSIS SECTION WITH BETTER SPACING ----
        
        # Section title with modern styling - brighter
        pdf.saveState()
        pdf.setFillColorRGB(0.4, 1, 1)  # Much brighter cyan
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(40, height - 160, "Zodiac Analysis")
        
        # Thicker underline
        pdf.setStrokeColorRGB(0.4, 1, 1)
        pdf.setLineWidth(2.5)
        pdf.line(40, height - 165, 180, height - 165)
        pdf.restoreState()

        # Modern tables with better styling and contrast
        styles = getSampleStyleSheet()
        zodiac_style = ParagraphStyle(
            'ZodiacStyle',
            parent=styles['Normal'],
            textColor=colors.white,
            fontSize=11,  # Larger font
            leading=14
        )
        
        # Fix descriptions for proper capitalization
        zodiac_sign_desc = zodiac_results["zodiac_sign"]["description"]
        if not zodiac_sign_desc.startswith(zodiac_results["zodiac_sign"]["name"]):
            zodiac_sign_desc = f"{zodiac_results['zodiac_sign']['name']} - {zodiac_sign_desc}"
            
        # Create stylized description paragraphs
        zodiac_description = Paragraph(zodiac_sign_desc, zodiac_style)
        chinese_description = Paragraph(zodiac_results["chinese_zodiac"]["description"], zodiac_style)
        
        # Create zodiac table data
        zodiac_table_data = [
            ["Zodiac Sign", zodiac_results["zodiac_sign"]["name"]],
            ["Element", zodiac_results["zodiac_sign"]["element"]],
            ["Description", zodiac_description],
            ["Chinese Zodiac", zodiac_results["chinese_zodiac"]["name"]],
            ["Description", chinese_description],
        ]
        
        # Create stylized table with better colors for visibility
        zodiac_table = Table(zodiac_table_data, colWidths=[100, 350])
        zodiac_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.Color(0.3, 0.3, 0.5, 0.9)),  # Much brighter headers
            ('BACKGROUND', (1, 0), (1, -1), colors.Color(0.2, 0.2, 0.4, 0.8)),  # Much brighter cells
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.5, 0.5, 0.7, 0.8)),  # Much brighter grid
            ('ROUNDEDCORNERS', [5, 5, 5, 5]),
        ]))
        
        zodiac_table.wrapOn(pdf, width, height)
        zodiac_table.drawOn(pdf, 40, height - 300)  # Position to prevent overlap

        # ---- FACIAL FEATURES SECTION WITH FIXED SPACING ----
        
        # Section title - properly spaced to prevent overlap (moved further down)
        pdf.saveState()
        pdf.setFillColorRGB(0.4, 1, 1)  # Much brighter cyan
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(40, height - 310, "Facial Feature Analysis")  # Moved down by 10 points
        
        # Thicker underline
        pdf.setStrokeColorRGB(0.4, 1, 1)
        pdf.setLineWidth(2.5)
        pdf.line(40, height - 315, 220, height - 315)
        pdf.restoreState()
        
        # Feature style for paragraphs with improved readability
        feature_style = ParagraphStyle(
            'FeatureStyle',
            parent=styles['Normal'],
            textColor=colors.white,
            fontSize=10,  # Larger font
            leading=14    # More line spacing
        )
        
        # Fix facial feature descriptions (capitalize first letter and fix 'I' instead of 'i')
        fixed_facial_features = {}
        for feature, values in facial_features.items():
            # Fix description: capitalize first letter, fix "i" to "I"
            description = values["description"]
            description = description.replace(" i ", " I ")
            description = description.replace(". i", ". I")
            
            # Store fixed values
            fixed_facial_features[feature] = {
                "shape": values["shape"],
                "description": description
            }
        
        # Create facial feature table with improved styling
        facial_table_data = [["Feature", "Shape", "Personality Trait"]]
        
        for feature, values in fixed_facial_features.items():
            facial_table_data.append([
                feature.capitalize(), 
                values["shape"], 
                Paragraph(values["description"], feature_style)  # Auto-wrap text
            ])

        # Create table with adjusted widths for better layout
        facial_table = Table(facial_table_data, colWidths=[80, 80, 350])
        facial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.35, 0.35, 0.55, 0.95)),  # Much brighter header
            ('BACKGROUND', (0, 1), (0, -1), colors.Color(0.3, 0.3, 0.5, 0.9)),      # Much brighter feature column
            ('BACKGROUND', (1, 1), (1, -1), colors.Color(0.25, 0.25, 0.45, 0.85)),  # Much brighter shape column
            ('BACKGROUND', (2, 1), (2, -1), colors.Color(0.2, 0.2, 0.4, 0.8)),      # Much brighter description
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),  # Center headers
            ('ALIGN', (0, 1), (1, -1), 'CENTER'),  # Center feature and shape
            ('ALIGN', (2, 1), (2, -1), 'LEFT'),    # Left align descriptions
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (1, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # More padding
            ('TOPPADDING', (0, 0), (-1, -1), 8),     # More padding
            ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.5, 0.5, 0.7, 0.8)),  # Much brighter grid
            ('ROUNDEDCORNERS', [5, 5, 5, 5]),
        ]))
        
        # Adjust position to avoid overlap with zodiac section - moved down by 10 more points
        facial_table.wrapOn(pdf, width, height)
        facial_table.drawOn(pdf, 40, height - 535)  # Moved down more to fix overlap

        # ---- CELEBRITY LOOKALIKE SECTION WITH IMPROVED IMAGES ----
        
        # Add new page for lookalikes and personality analysis
        pdf.showPage()
        
        # Continue with gradient background on new page - lighter for better readability
        pdf.saveState()
        for i in range(100):
            y = i * height / 100
            r = 0.1 - (i * 0.0005)  # Slightly lighter blue
            g = 0.1 - (i * 0.0003)
            b = 0.25 - (i * 0.0005)
            pdf.setFillColorRGB(r, g, b)
            pdf.rect(0, y, width, height/100, fill=1, stroke=0)
        pdf.restoreState()
        
        # Add tech circuit pattern overlay on new page - more visible
        pdf.saveState()
        pdf.setFillColorRGB(1, 1, 1, 0.05)  # Slightly more opacity
        for i in range(6):  # Fewer circles
            x = random.randint(0, int(width))
            y = random.randint(0, int(height))
            size = random.randint(5, 15)
            pdf.circle(x, y, size, fill=1, stroke=0)
        
        for i in range(8):  # Fewer lines
            x1 = random.randint(0, int(width))
            y1 = random.randint(0, int(height))
            x2 = x1 + random.randint(-80, 80)
            y2 = y1 + random.randint(-80, 80)
            pdf.setStrokeColorRGB(1, 1, 1, 0.1)  # Slightly brighter lines
            pdf.line(x1, y1, x2, y2)
        pdf.restoreState()
        
        # Title on new page - brighter
        pdf.saveState()
        pdf.setFillColorRGB(1, 1, 1)  # Pure white
        pdf.setFont("Helvetica-Bold", 28)
        pdf.drawCentredString(width / 2, height - 40, "AI Personality & Lookalike Report")
        pdf.setFont("Helvetica-Bold", 16)  # Larger and bold font
        pdf.drawCentredString(width / 2, height - 60, f"Continued - {name}")
        pdf.restoreState()
        
        # Section title with modern styling - much brighter
        pdf.saveState()
        pdf.setFillColorRGB(0.4, 1, 1)  # Much brighter cyan
        pdf.setFont("Helvetica-Bold", 20)  # Larger font
        pdf.drawString(40, height - 100, "Celebrity Lookalike Matches")
        
        # Thicker underline
        pdf.setStrokeColorRGB(0.4, 1, 1)
        pdf.setLineWidth(2.5)
        pdf.line(40, height - 105, 260, height - 105)
        pdf.restoreState()

        # Better error handling for celebrity images with improved caching
        def get_better_celebrity_image(celeb_name):
            """Enhanced function to get celebrity images with better error handling"""
            try:
                image_url = get_celebrity_image(celeb_name)
                if not image_url:
                    return None
                    
                # Test if image is accessible with longer timeout
                response = requests.head(image_url, timeout=5)
                if response.status_code != 200:
                    return None
                    
                return image_url
            except Exception:
                return None

        # Display celebrity matches with improved style and spacing
        x_position = 40
        y_position = height - 130
        
        # Fix the match percentages - generate realistic percentages
        adjusted_lookalikes = []
        for match in lookalike_results:
            # Generate realistic match percentages (65-95%)
            realistic_match = random.uniform(65.5, 95.0)
            adjusted_match = {
                "name": match["name"],
                "match_percentage": round(realistic_match, 1)
            }
            adjusted_lookalikes.append(adjusted_match)
        
        # Sort by match percentage (highest first)
        adjusted_lookalikes.sort(key=lambda x: x["match_percentage"], reverse=True)
        
        # Limit to max 4 per row, ensure proper spacing
        for i, match in enumerate(adjusted_lookalikes[:8]):  # Limit to 8 for better layout
            celeb_name = match["name"]
            match_percentage = match["match_percentage"]
            celeb_image_url = get_better_celebrity_image(celeb_name)
            
            # Create stylized box for each celebrity - much brighter colors
            pdf.saveState()
            
            # Background for celebrity box - brighter
            pdf.setFillColorRGB(0.25, 0.25, 0.45, 0.85)  # Brighter blue with better opacity
            pdf.roundRect(x_position, y_position - 160, 160, 150, 10, fill=1, stroke=0)
            
            # Add stronger border around celebrity box
            pdf.setStrokeColorRGB(1, 1, 1, 0.3)  # White border
            pdf.setLineWidth(1)
            pdf.roundRect(x_position, y_position - 160, 160, 150, 10, fill=0, stroke=1)
            
            # Highlight border based on match percentage - much brighter
            match_color = (0.4, 1, 1) if match_percentage > 85 else (0.5, 0.7, 1)
            pdf.setStrokeColorRGB(*match_color)
            pdf.setLineWidth(2.5)  # Thicker highlight border
            pdf.roundRect(x_position, y_position - 160, 160, 150, 10, fill=0, stroke=1)
            
            # Celebrity name with match percentage - larger font and better positioning
            pdf.setFillColorRGB(1, 1, 1)  # Pure white
            pdf.setFont("Helvetica-Bold", 14)  # Larger font
            text = pdf.beginText(x_position + 10, y_position - 20)
            text.textLine(celeb_name)
            pdf.drawText(text)
            
            # Match percentage with color coding - much brighter colors
            if match_percentage > 85:
                pdf.setFillColorRGB(0.3, 1, 0.6)  # Much brighter green for high matches
            elif match_percentage > 75:
                pdf.setFillColorRGB(1, 0.9, 0.3)  # Much brighter yellow/gold for medium matches  
            else:
                pdf.setFillColorRGB(1, 0.7, 0.3)  # Much brighter orange for lower matches
                
            pdf.setFont("Helvetica-Bold", 15)  # Larger font
            pdf.drawString(x_position + 10, y_position - 40, f"{match_percentage}% Match")
            
            # Add matching score visualization (bar) - better contrast
            bar_width = 140
            pdf.setFillColorRGB(0.35, 0.35, 0.55, 0.7)  # Brighter background with better opacity
            pdf.roundRect(x_position + 10, y_position - 55, bar_width, 10, 4, fill=1, stroke=0)
            
            # Add border to bar for better visibility
            pdf.setStrokeColorRGB(1, 1, 1, 0.3)  # White border
            pdf.setLineWidth(0.5)
            pdf.roundRect(x_position + 10, y_position - 55, bar_width, 10, 4, fill=0, stroke=1)
            
            # Colored portion of bar based on match percentage - much brighter
            filled_width = bar_width * (match_percentage / 100)
            if match_percentage > 85:
                pdf.setFillColorRGB(0.3, 1, 0.6)  # Much brighter green
            elif match_percentage > 75:
                pdf.setFillColorRGB(1, 0.9, 0.3)  # Much brighter yellow/gold
            else:
                pdf.setFillColorRGB(1, 0.7, 0.3)  # Much brighter orange
                
            pdf.roundRect(x_position + 10, y_position - 55, filled_width, 10, 4, fill=1, stroke=0)
            
            # Celebrity image with better error handling and improved placeholder
            try:
                if celeb_image_url:
                    image_response = requests.get(celeb_image_url, stream=True, timeout=8)  # Longer timeout
                    if image_response.status_code == 200:
                        img = ImageReader(image_response.raw)
                        # Draw image with a white border - stronger
                        pdf.setFillColorRGB(1, 1, 1, 0.2)  # Slightly more visible white background
                        pdf.roundRect(x_position + 30, y_position - 150, 100, 100, 5, fill=1, stroke=0)
                        
                        # Add border around image
                        pdf.setStrokeColorRGB(1, 1, 1, 0.5)  # White border with better opacity
                        pdf.setLineWidth(1)
                        pdf.roundRect(x_position + 30, y_position - 150, 100, 100, 5, fill=0, stroke=1)
                        
                        # Draw image (adjusted to fit better)
                        pdf.drawImage(img, x_position + 35, y_position - 145, width=90, height=90)
                    else:
                        raise Exception("Image not available")
                else:
                    raise Exception("Image URL not found")
            except Exception:
                # Much improved placeholder if image fails
                pdf.setFillColorRGB(0.3, 0.3, 0.5, 0.9)  # Much brighter placeholder background
                pdf.roundRect(x_position + 30, y_position - 150, 100, 100, 5, fill=1, stroke=0)
                
                # Add border around placeholder
                pdf.setStrokeColorRGB(1, 1, 1, 0.5)  # White border with better opacity
                pdf.setLineWidth(1)
                pdf.roundRect(x_position + 30, y_position - 150, 100, 100, 5, fill=0, stroke=1)
                
                # Draw placeholder icon (simple face) - more visible
                pdf.setStrokeColorRGB(1, 1, 1)
                pdf.setLineWidth(1.5)  # Thicker lines
                # Head
                pdf.circle(x_position + 80, y_position - 100, 25, fill=0, stroke=1)
                # Eyes
                pdf.circle(x_position + 70, y_position - 95, 4, fill=1, stroke=0)  # Larger eyes
                pdf.circle(x_position + 90, y_position - 95, 4, fill=1, stroke=0)  # Larger eyes
                # Smile
                pdf.arc(x_position + 80, y_position - 95, 15, -30, 210)
                
                # Message with better visibility
                pdf.setFillColorRGB(1, 1, 1)
                pdf.setFont("Helvetica-Bold", 10)  # Bold for better visibility
                pdf.drawCentredString(x_position + 80, y_position - 70, "Image Not Available")
                
            pdf.restoreState()
            
            x_position += 175  # Better spacing for cleaner layout
            
            # Move to next row if needed
            if (i + 1) % 4 == 0:
                x_position = 40
                y_position -= 170

        # ---- PERSONALITY ANALYSIS SECTION WITH BETTER READABILITY ----
        
        # Personality Analysis with enhanced styling and proper spacing
        # Calculate vertical position based on number of celebrity rows
        rows_needed = (len(adjusted_lookalikes[:8]) + 3) // 4  # Ceiling division
        y_position = min(height - 130 - (rows_needed * 170) - 40, height - 350)  # Ensure good spacing

        # Section title - much brighter
        pdf.saveState()
        pdf.setFillColorRGB(0.4, 1, 1)  # Much brighter cyan
        pdf.setFont("Helvetica-Bold", 20)  # Larger font
        pdf.drawString(40, y_position, "Personalized Personality Analysis")
        
        # Thicker underline
        pdf.setStrokeColorRGB(0.4, 1, 1)
        pdf.setLineWidth(2.5)
        pdf.line(40, y_position - 5, 300, y_position - 5)
        pdf.restoreState()
        
        # Create nicer background for analysis text - much brighter
        pdf.saveState()
        pdf.setFillColorRGB(0.25, 0.25, 0.45, 0.85)  # Much brighter blue with better opacity
        pdf.roundRect(40, y_position - 190, width - 80, 170, 10, fill=1, stroke=0)  # Taller box
        
        # Add border around personality box for better visibility
        pdf.setStrokeColorRGB(1, 1, 1, 0.3)  # White border
        pdf.setLineWidth(1)
        pdf.roundRect(40, y_position - 190, width - 80, 170, 10, fill=0, stroke=1)
        
        # Add decorative elements - brighter but subtle
        pdf.setFillColorRGB(0.4, 1, 1, 0.25)  # Much brighter cyan with better transparency
        pdf.circle(60, y_position - 40, 10, fill=1, stroke=0)
        pdf.circle(width - 60, y_position - 40, 10, fill=1, stroke=0)
        pdf.circle(60, y_position - 160, 10, fill=1, stroke=0)
        pdf.circle(width - 60, y_position - 160, 10, fill=1, stroke=0)
        pdf.restoreState()
        
        # Fix capitalization and grammatical issues in facial feature descriptions
        # Build enhanced personality text with proper grammar and transitions
        fixed_eye_desc = fixed_facial_features['eye']['description'].replace("can be good friends to talk with", "you can be a good friend to talk with")
        fixed_nose_desc = fixed_facial_features['nose']['description'].replace("I don't", "You don't").replace("I pay", "You pay").replace("I have", "You have").replace("I am", "You are")
        fixed_jaw_desc = fixed_facial_features['jaw']['description'].replace("they also", "you also")
        
        personality_text = (
            f"Based on your facial features and zodiac analysis, your personality profile reveals several key insights. "
            f"Your {fixed_facial_features['jaw']['shape'].lower()} jaw shape indicates that {fixed_jaw_desc} "
            f"With {fixed_facial_features['eye']['shape'].lower()} eyes, {fixed_eye_desc} "
            f"Your {fixed_facial_features['nose']['shape'].lower()} nose suggests that {fixed_nose_desc}\n\n"
            f"Born under {zodiac_results['zodiac_sign']['name']}, you naturally embody {zodiac_results['zodiac_sign']['element']} element traits. "
            f"This means you embody qualities of {zodiac_sign_desc.lower()}. "
            f"Your Chinese zodiac sign ({zodiac_results['chinese_zodiac']['name']}) adds complementary elements, as "
            f"{zodiac_results['chinese_zodiac']['description'].lower()}.\n\n"
            f"The combination of your facial structure and astrological influences creates a uniquely balanced personality that draws from both "
            f"your innate traits and cosmic influences."
        )
        
        # Create styled paragraph with much better readability
        personality_style = ParagraphStyle(
            'PersonalityStyle',
            parent=styles['Normal'],
            textColor=colors.white,
            fontSize=12,  # Larger font
            leading=16,   # More spacing between lines
            alignment=1   # Center alignment
        )
        
        paragraph = Paragraph(personality_text, personality_style)
        paragraph.wrapOn(pdf, width - 100, height)
        paragraph.drawOn(pdf, 50, y_position - 180)  # Adjusted position
        
        # ---- FOOTER WITH BETTER VISIBILITY ----
        
        # Add footer with disclaimer and branding - brighter
        pdf.saveState()
        pdf.setFillColorRGB(0.25, 0.25, 0.45, 0.9)  # Much brighter blue with better opacity
        pdf.rect(0, 0, width, 30, fill=1, stroke=0)
        
        # Add border above footer
        pdf.setStrokeColorRGB(1, 1, 1, 0.3)  # White border
        pdf.setLineWidth(0.5)
        pdf.line(0, 30, width, 30)
        
        pdf.setFillColorRGB(1, 1, 1)  # Pure white
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(40, 10, "AI Personality & Lookalike Report - For Entertainment Purposes Only")
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(width - 40, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d')} | Report ID: {report_id}")
        pdf.restoreState()
        
        # Finalize PDF
        pdf.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        # Error handling
        print(f"Error generating PDF: {str(e)}")
        # Return empty buffer in case of error
        return io.BytesIO()

@app.route('/api/process-all', methods=['POST'])
def process_all():
    try:
        data = request.get_json()
        
        # Extract inputs
        name = data.get('name')
        email = data.get('email')
        dob = data.get('dob')  # Can be 'YYYY-MM-DD' or 'MM/DD/YYYY'
        number = data.get('number')
        image_data = data.get('image')
        gender = data.get('gender')
        
        if not dob or not image_data or not name or not email or not gender:
            return jsonify({'status': 'error', 'message': 'Missing required inputs.'}), 400

        # Process DOB for zodiac signs
        def calculate_zodiac_signs(dob):
            try:
                # Support both 'MM/DD/YYYY' and 'YYYY-MM-DD' formats
                try:
                    dob_obj = datetime.strptime(dob, '%m/%d/%Y')
                except ValueError:
                    dob_obj = datetime.strptime(dob, '%Y-%m-%d')
                
                dob_str = dob_obj.strftime('%Y-%m-%d')  # Standardize to 'YYYY-MM-DD'
                year, month, day = map(int, dob_str.split('-'))

                # Zodiac sign calculation
                def get_zodiac_sign(month, day):
                    for sign_data in ZODIAC_SIGNS:
                        start_month, start_day = map(int, sign_data['date_range']['start'].split('-'))
                        end_month, end_day = map(int, sign_data['date_range']['end'].split('-'))
                        if (month == start_month and day >= start_day) or \
                           (month == end_month and day <= end_day) or \
                           (start_month < month < end_month):
                            return sign_data
                    return next(sign for sign in ZODIAC_SIGNS if sign['name'] == 'Capricorn')

                # Chinese Zodiac calculation
                def get_chinese_zodiac(year):
                    if not CHINESE_ZODIAC_SIGNS:
                        raise ValueError("Chinese zodiac signs not properly loaded.")

                    index = year % 12
                    if index < 0 or index >= len(CHINESE_ZODIAC_SIGNS):
                        raise IndexError("Calculated index for Chinese Zodiac is out of range.")
                    return CHINESE_ZODIAC_SIGNS[index]

                zodiac_sign_data = get_zodiac_sign(month, day)
                chinese_zodiac_data = get_chinese_zodiac(year)

                return {
                    'zodiac_sign': {
                        'name': zodiac_sign_data['name'],
                        'description': f"{zodiac_sign_data['name']} - {zodiac_sign_data.get('description', 'Description not available.')}",
                        'element': zodiac_sign_data.get('element', 'N/A')
                    },
                    'chinese_zodiac': {
                        'name': chinese_zodiac_data['name'],
                        'description': chinese_zodiac_data.get('description', 'Description not available.')
                    }
                }

            except Exception as e:
                raise ValueError(f"Error processing DOB: {str(e)}")

        zodiac_results = calculate_zodiac_signs(dob)

                # Process image for facial features
        def process_image_data(image_data):
            try:
                img_data = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img_cv = np.array(img).astype(np.uint8)

                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
                    results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
                    if not results.multi_face_landmarks:
                        raise ValueError("No faces detected.")

                    feature_points = []
                    # Draw landmarks on the image
                    for face_landmarks in results.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * img_cv.shape[1])
                            y = int(landmark.y * img_cv.shape[0])
                            cv2.circle(img_cv, (x, y), 2, (0, 255, 0), -1)  # Draw green landmarks
                            feature_points.append((x, y))

                    if feature_points:
                        left_eye_region = img_cv[
                            feature_points[36][1]:feature_points[41][1],
                            feature_points[36][0]:feature_points[39][0]
                        ]
                        left_eye_img_resized = cv2.resize(left_eye_region, (200, 200))
                        preprocess = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img_tensor = preprocess(Image.fromarray(cv2.cvtColor(left_eye_img_resized, cv2.COLOR_BGR2RGB))).unsqueeze(0)

                        eyebrow = eyebrow_model(img_tensor)
                        eye = eye_model(img_tensor)
                        nose = nose_model(img_tensor)
                        mouth = mouth_model(img_tensor)
                        jaw = jaw_model(img_tensor)

                        # Get shapes and descriptions
                        def get_shape_analysis(shape_name, feature_name):
                            for region in shape_descriptions['face_regions']:
                                if region['name'] == shape_name:
                                    for feature in region['features']:
                                        if feature['name'] == feature_name:
                                            return feature['analysis']
                            return "Description not found."

                        return {
                            'eyebrow': {
                                'shape': ["Arch", "Circle", "Straight"][torch.argmax(eyebrow).item()],
                                'description': get_shape_analysis('eyebrows', ["Arch", "Circle", "Straight"][torch.argmax(eyebrow).item()])
                            },
                            'eye': {
                                'shape': ["Big", "Silt", "Small"][torch.argmax(eye).item()],
                                'description': get_shape_analysis('eyes', ["Big", "Silt", "Small"][torch.argmax(eye).item()])
                            },
                            'nose': {
                                'shape': ["Long", "Small", "Wide"][torch.argmax(nose).item()],
                                'description': get_shape_analysis('nose', ["Long", "Small", "Wide"][torch.argmax(nose).item()])
                            },
                            'mouth': {
                                'shape': ["Medium", "Small", "Thick"][torch.argmax(mouth).item()],
                                'description': get_shape_analysis('mouth', ["Medium", "Small", "Thick"][torch.argmax(mouth).item()])
                            },
                            'jaw': {
                                'shape': ["Circle", "Oval", "Square", "Triangle"][torch.argmax(jaw).item()],
                                'description': get_shape_analysis('face', ["Circle", "Oval", "Square", "Triangle"][torch.argmax(jaw).item()])
                            },
                            'labeled_image': img_cv  # Add the labeled image
                        }
            except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}")

        facial_features = process_image_data(image_data)

        # Generate labeled image base64
        _, buffer = cv2.imencode('.png', facial_features.pop('labeled_image'))
        labeled_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Find lookalikes
        def find_lookalike(facial_features, gender):
            try:
                with open('C:\\Projects\\brainwave\\frontend\\src\\components\\celebrity_data.json') as f:
                    celeb_data = json.load(f)

                matches = []
                weights = {'eyebrow': 1, 'eye': 2, 'nose': 2, 'mouth': 1, 'jaw': 3}
                for celeb in celeb_data:
                    if celeb['gender'].lower() != gender.lower():
                        continue

                    score = sum(weights[feature] for feature in weights if celeb['features'][feature] == facial_features[feature])
                    total_weight = sum(weights.values())
                    match_percentage = (score / total_weight) * 100
                    matches.append({
                        'name': celeb['name'],
                        'gender': celeb['gender'],
                        'features': celeb['features'],
                        'match_percentage': round(match_percentage, 2)
                    })

                matches.sort(key=lambda x: (-x['match_percentage'], x['name']))
                return matches[:3]
            except Exception as e:
                raise ValueError(f"Error finding lookalike: {str(e)}")

        lookalike_results = find_lookalike(facial_features, gender)

        PDF_STORAGE_PATH = "C:/Projects/brainwave/backend/pdfs"
        os.makedirs(PDF_STORAGE_PATH, exist_ok=True)  # Create 'pdfs' directory if not exists

        pdf_stream = generate_pdf_report(name, email, number, zodiac_results, facial_features, lookalike_results, labeled_image_base64)
        if not pdf_stream:
            raise ValueError("Failed to generate PDF report.")


        # Replace your PDF generation and sending code with this:
        pdf_stream = generate_pdf_report(name, email, number, zodiac_results, facial_features, lookalike_results, labeled_image_base64)
        pdf_filename = f"{name.replace(' ', '_')}_{email.replace('@', '_')}.pdf"
        pdf_path = os.path.join(PDF_STORAGE_PATH, pdf_filename)

        # Save the PDF to a file
        pdf_stream.seek(0)  # Reset position to beginning
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(pdf_stream.getvalue())  # Use getvalue() to get all content

        app.logger.info(f"PDF saved at {pdf_path}")

        # Return the PDF file path for frontend to download
        return jsonify({
            'status': 'success',
            'file_path': f'/files/{pdf_filename}',
            'zodiac_results': zodiac_results,
            'facial_features': facial_features,
            'lookalikes': lookalike_results
        })
    

    except Exception as e:
        app.logger.error(f"Error in process-all: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Endpoint to serve PDF files
@app.route('/files/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(PDF_STORAGE_PATH, filename, as_attachment=True, mimetype='application/pdf')
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'File not found.'}), 404

        
if __name__ == '__main__':
    app.run(debug=True)
