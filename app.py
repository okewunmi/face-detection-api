from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import time
import subprocess
import sys

app = Flask(__name__)
CORS(app)

# Install face_recognition library (uses dlib, 128-dimensional)
try:
    import face_recognition
except ImportError:
    print("Installing face_recognition...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "face_recognition"])
    import face_recognition

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'success': True,
        'message': 'Face Detection API is running (face_recognition - 128D)',
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
        
        print('📸 Extracting face with face_recognition...')
        start = time.time()
        
        # Decode image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'message': 'Failed to decode image'}), 400
        
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]
        print(f'📐 Image size: {width}x{height}')
        
        # Detect faces and get 128-dimensional encodings
        face_locations = face_recognition.face_locations(rgb_img, model='hog')
        
        if len(face_locations) == 0:
            # Try with CNN model (more accurate but slower)
            print('🔍 Trying CNN model...')
            face_locations = face_recognition.face_locations(rgb_img, model='cnn')
        
        if len(face_locations) == 0:
            print(f'❌ No face detected')
            brightness = int(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
            
            return jsonify({
                'success': False, 
                'message': 'No face detected. Please ensure face is clearly visible and well-lit.',
                'debug': {
                    'imageSize': f'{width}x{height}',
                    'brightness': brightness
                }
            }), 400
        
        # Get face encodings (128-dimensional descriptors)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({
                'success': False,
                'message': 'Face detected but could not extract features'
            }), 400
        
        # Use the first face
        descriptor = face_encodings[0].tolist()
        processing_time = int((time.time() - start) * 1000)
        
        # Calculate face area
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top
        face_area_ratio = (face_width * face_height) / (width * height)
        
        print(f'✅ Face extracted in {processing_time}ms')
        print(f'📊 Descriptor dimension: {len(descriptor)}')
        print(f'📊 Face area: {face_area_ratio*100:.1f}% of image')
        
        return jsonify({
            'success': True,
            'descriptor': descriptor,
            'confidence': 95,
            'processingTime': processing_time,
            'debug': {
                'faceCount': len(face_locations),
                'faceArea': f'{face_area_ratio*100:.1f}%',
                'imageSize': f'{width}x{height}',
                'descriptorDim': len(descriptor)
            }
        })
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/face/verify', methods=['POST'])
def verify_face():
    try:
        data = request.get_json()
        
        input_desc = np.array(data['inputDescriptor'])
        students = data['students']
        
        if not students:
            return jsonify({'success': True, 'matched': False, 'message': 'No students'})
        
        print(f'🔍 Verifying against {len(students)} students...')
        print(f'📊 Input descriptor dimension: {len(input_desc)}')
        start = time.time()
        
        # Verify descriptor dimensions
        if len(input_desc) != 128:
            return jsonify({
                'success': False,
                'message': f'Invalid descriptor dimension: {len(input_desc)} (expected 128)'
            }), 400
        
        best_match = None
        best_distance = float('inf')
        
        for student in students:
            student_desc = np.array(student['descriptor'])
            
            # Verify student descriptor dimension
            if len(student_desc) != 128:
                print(f'⚠️ Skipping {student.get("matricNumber", "unknown")}: wrong dimension {len(student_desc)}')
                continue
            
            # Calculate Euclidean distance (face_recognition uses this)
            distance = np.linalg.norm(input_desc - student_desc)
            
            if distance < best_distance:
                best_distance = distance
                best_match = student
        
        processing_time = int((time.time() - start) * 1000)
        
        # face_recognition typically uses threshold of 0.6
        threshold = 0.6
        matched = best_distance < threshold
        
        if not matched:
            print(f'❌ No match found (best distance: {best_distance:.3f})')
            return jsonify({
                'success': True,
                'matched': False,
                'message': 'No match found',
                'bestDistance': float(best_distance),
                'processingTime': processing_time
            })
        
        # Convert distance to confidence percentage (inverse relationship)
        # Distance ranges from 0 (perfect match) to ~1.0
        confidence = int(max(0, min(100, (1 - best_distance) * 100)))
        
        print(f'✅ Match: {best_match["matricNumber"]} (confidence: {confidence}%, distance: {best_distance:.3f})')
        
        return jsonify({
            'success': True,
            'matched': True,
            'student': best_match,
            'confidence': confidence,
            'distance': float(best_distance),
            'processingTime': processing_time
        })
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    print('🚀 Starting Face Recognition API (128-dimensional)')
    app.run(host='0.0.0.0', port=5000, debug=True)