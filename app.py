from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import base64
import os
import time

app = Flask(__name__)
CORS(app)

# Create temp directory
os.makedirs('/tmp/faces', exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'success': True,
        'message': 'Face Detection API (DeepFace)',
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
            return jsonify({'success': False, 'message': 'Image required'}), 400
        
        print('📸 Extracting face...')
        start = time.time()
        
        # Decode base64
        img_data = base64.b64decode(data['image'].split(',')[1])
        temp_path = f'/tmp/faces/temp_{int(time.time() * 1000)}.jpg'
        
        with open(temp_path, 'wb') as f:
            f.write(img_data)
        
        # Extract embedding
        embeddings = DeepFace.represent(
            img_path=temp_path,
            model_name='Facenet512',
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        # Cleanup
        os.remove(temp_path)
        
        if not embeddings:
            return jsonify({'success': False, 'message': 'No face detected'}), 400
        
        descriptor = embeddings[0]['embedding']
        processing_time = int((time.time() - start) * 1000)
        
        print(f'✅ Extracted in {processing_time}ms')
        
        return jsonify({
            'success': True,
            'descriptor': descriptor,
            'confidence': 95,
            'processingTime': processing_time
        })
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        # Cleanup on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/face/verify', methods=['POST'])
def verify_face():
    try:
        data = request.get_json()
        
        input_desc = np.array(data['inputDescriptor'])
        students = data['students']
        
        if not students:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No students'
            })
        
        print(f'🔍 Verifying against {len(students)} students...')
        start = time.time()
        
        # Cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        best_match = None
        best_similarity = -1
        
        for student in students:
            student_desc = np.array(student['descriptor'])
            similarity = cosine_similarity(input_desc, student_desc)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student
        
        processing_time = int((time.time() - start) * 1000)
        
        # Threshold for Facenet512 is ~0.4 similarity
        threshold = 0.4
        matched = best_similarity > threshold
        
        if not matched:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No match found',
                'bestDistance': float(1 - best_similarity),
                'processingTime': processing_time
            })
        
        confidence = int(best_similarity * 100)
        
        print(f'✅ Match: {best_match["matricNumber"]} ({confidence}%)')
        
        return jsonify({
            'success': True,
            'matched': True,
            'student': best_match,
            'confidence': confidence,
            'distance': float(1 - best_similarity),
            'processingTime': processing_time
        })
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)