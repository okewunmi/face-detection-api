from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
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
        'message': 'Face Detection API is running',
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
        base64_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert RGBA to RGB
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
        
        # Find faces
        face_locations = face_recognition.face_locations(image_array)
        
        if len(face_locations) == 0:
            return jsonify({'success': False, 'message': 'No face detected'}), 400
        
        # Extract encodings
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({'success': False, 'message': 'Could not extract face'}), 400
        
        descriptor = face_encodings[0].tolist()
        
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
        
        input_descriptor = np.array(data['inputDescriptor'])
        students = data['students']
        
        if len(students) == 0:
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No students'
            })
        
        print(f'🔍 Verifying against {len(students)} students...')
        start = time.time()
        
        # Prepare known encodings
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
                'message': 'No valid student descriptors'
            })
        
        # Calculate distances
        face_distances = face_recognition.face_distance(known_encodings, input_descriptor)
        
        # Find best match
        best_match_index = np.argmin(face_distances)
        best_distance = float(face_distances[best_match_index])
        
        threshold = 0.6
        matched = best_distance < threshold
        
        processing_time = int((time.time() - start) * 1000)
        
        if not matched:
            print(f'❌ No match (distance: {best_distance:.3f})')
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No match found',
                'bestDistance': best_distance,
                'processingTime': processing_time
            })
        
        matched_student = student_map[best_match_index]
        confidence = int((1 - best_distance) * 100)
        
        print(f'✅ Match: {matched_student["matricNumber"]} ({confidence}%)')
        
        return jsonify({
            'success': True,
            'matched': True,
            'student': matched_student,
            'confidence': confidence,
            'distance': best_distance,
            'processingTime': processing_time
        })
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)