from flask import Flask, request, jsonify
from flask_cors import CORS
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import base64
import cv2
import os
import time

app = Flask(__name__)
CORS(app)

# Initialize face analyzer
face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'success': True,
        'message': 'Face Detection API (InsightFace)',
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
        
        # Decode base64 to numpy array
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get face analyzer
        analyzer = get_face_app()
        
        # Detect faces
        faces = analyzer.get(img)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'}), 400
        
        # Get embedding from first face
        descriptor = faces[0].embedding.tolist()
        
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
        
        # Compute cosine similarity
        def compute_sim(emb1, emb2):
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        best_match = None
        best_similarity = -1
        
        for student in students:
            student_desc = np.array(student['descriptor'])
            similarity = compute_sim(input_desc, student_desc)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student
        
        processing_time = int((time.time() - start) * 1000)
        
        # Threshold for InsightFace is typically 0.3-0.4
        threshold = 0.35
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