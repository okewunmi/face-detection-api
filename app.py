from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import base64
import io
from PIL import Image
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'success': True,
        'message': 'Face Detection API is running',
        'endpoints': {
            'extract': '/api/face/extract (POST)',
            'verify': '/api/face/verify (POST)'
        }
    })

@app.route('/api/face/extract', methods=['POST'])
def extract_face():
    """Extract face descriptor from base64 image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'Image is required in request body'
            }), 400
        
        image_data = data['image']
        
        if not image_data.startswith('data:image/'):
            return jsonify({
                'success': False,
                'message': 'Invalid image format. Must be base64 data URI'
            }), 400
        
        print('📸 Extracting face descriptor...')
        start_time = time.time()
        
        # Decode base64 image
        base64_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert RGBA to RGB if necessary
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
        
        # Find face locations
        face_locations = face_recognition.face_locations(image_array)
        
        if len(face_locations) == 0:
            print('❌ No face detected')
            return jsonify({
                'success': False,
                'message': 'No face detected in image'
            }), 400
        
        # Extract face encodings (descriptors)
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) == 0:
            print('❌ No face encoding extracted')
            return jsonify({
                'success': False,
                'message': 'Could not extract face descriptor'
            }), 400
        
        # Use the first face found
        descriptor = face_encodings[0].tolist()
        
        processing_time = int((time.time() - start_time) * 1000)
        print(f'✅ Face extracted successfully in {processing_time}ms')
        
        return jsonify({
            'success': True,
            'descriptor': descriptor,
            'confidence': 95,  # face_recognition doesn't provide confidence scores
            'processingTime': processing_time
        }), 200
        
    except Exception as e:
        print(f'❌ Extract API error: {str(e)}')
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/face/verify', methods=['POST'])
def verify_face():
    """Verify face against database of students"""
    try:
        data = request.get_json()
        
        if not data or 'inputDescriptor' not in data:
            return jsonify({
                'success': False,
                'message': 'inputDescriptor is required'
            }), 400
        
        if 'students' not in data or not isinstance(data['students'], list):
            return jsonify({
                'success': False,
                'message': 'students array is required'
            }), 400
        
        input_descriptor = np.array(data['inputDescriptor'])
        students = data['students']
        
        if len(students) == 0:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No students in database to compare against'
            }), 200
        
        print(f'🔍 Verifying face against {len(students)} students...')
        start_time = time.time()
        
        # Prepare known face encodings
        known_encodings = []
        student_map = {}
        
        for student in students:
            if 'descriptor' in student and 'matricNumber' in student:
                encoding = np.array(student['descriptor'])
                known_encodings.append(encoding)
                student_map[len(known_encodings) - 1] = student
        
        if len(known_encodings) == 0:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No valid student descriptors found'
            }), 200
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_encodings, input_descriptor)
        
        # Find best match
        best_match_index = np.argmin(face_distances)
        best_distance = float(face_distances[best_match_index])
        
        # Threshold: 0.6 (lower is better match)
        threshold = 0.6
        matched = best_distance < threshold
        
        processing_time = int((time.time() - start_time) * 1000)
        print(f'⏱️ Verification time: {processing_time}ms')
        
        if not matched:
            print(f'❌ No match found (best distance: {best_distance:.3f})')
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No matching student found',
                'bestDistance': best_distance,
                'processingTime': processing_time
            }), 200
        
        matched_student = student_map[best_match_index]
        confidence = int((1 - best_distance) * 100)
        
        print(f'✅ Match found: {matched_student["matricNumber"]} (confidence: {confidence}%)')
        
        return jsonify({
            'success': True,
            'matched': True,
            'student': matched_student,
            'confidence': confidence,
            'distance': best_distance,
            'processingTime': processing_time
        }), 200
        
    except Exception as e:
        print(f'❌ Verify API error: {str(e)}')
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # For development
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### Step 4: `.gitignore`
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv
*.log
.DS_Store