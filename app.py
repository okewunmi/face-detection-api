from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import base64
import io
from PIL import Image
import time

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'success': True,
        'message': 'Face Detection API is running (DeepFace)',
        'endpoints': {
            'extract': '/api/face/extract (POST)',
            'verify': '/api/face/verify (POST)'
        }
    })

@app.route('/api/face/extract', methods=['POST'])
def extract_face():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'Image is required'
            }), 400
        
        print('📸 Extracting face descriptor...')
        start_time = time.time()
        
        # Decode base64 image
        base64_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        
        # Save temporarily
        temp_path = '/tmp/temp_image.jpg'
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        # Extract embedding using DeepFace
        embedding_objs = DeepFace.represent(
            img_path=temp_path,
            model_name='Facenet',
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        if not embedding_objs:
            return jsonify({
                'success': False,
                'message': 'No face detected'
            }), 400
        
        descriptor = embedding_objs[0]['embedding']
        
        processing_time = int((time.time() - start_time) * 1000)
        print(f'✅ Face extracted in {processing_time}ms')
        
        return jsonify({
            'success': True,
            'descriptor': descriptor,
            'confidence': 95,
            'processingTime': processing_time
        }), 200
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/face/verify', methods=['POST'])
def verify_face():
    try:
        data = request.get_json()
        
        input_descriptor = np.array(data['inputDescriptor'])
        students = data['students']
        
        if len(students) == 0:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No students to compare'
            }), 200
        
        print(f'🔍 Verifying against {len(students)} students...')
        start_time = time.time()
        
        # Calculate cosine similarity
        def cosine_distance(a, b):
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        best_match = None
        best_distance = float('inf')
        
        for student in students:
            student_desc = np.array(student['descriptor'])
            distance = cosine_distance(input_descriptor, student_desc)
            
            if distance < best_distance:
                best_distance = distance
                best_match = student
        
        processing_time = int((time.time() - start_time) * 1000)
        
        threshold = 0.4  # DeepFace threshold
        matched = best_distance < threshold
        
        if not matched:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No match found',
                'bestDistance': float(best_distance),
                'processingTime': processing_time
            }), 200
        
        confidence = int((1 - best_distance) * 100)
        
        print(f'✅ Match: {best_match["matricNumber"]} ({confidence}%)')
        
        return jsonify({
            'success': True,
            'matched': True,
            'student': best_match,
            'confidence': confidence,
            'distance': float(best_distance),
            'processingTime': processing_time
        }), 200
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)