from flask import Flask, request, jsonify
from flask_cors import CORS
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import base64
import cv2
import time

app = Flask(__name__)
CORS(app)

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
        'message': 'Face Detection API is running',
        'endpoints': {
            'extract': '/api/face/extract (POST)',
            'verify': '/api/face/verify (POST)'
        }
    })

# @app.route('/api/face/extract', methods=['POST'])
# def extract_face():
#     try:
#         data = request.get_json()
        
#         if not data or 'image' not in data:
#             return jsonify({'success': False, 'message': 'Image required'}), 400
        
#         print('📸 Extracting face...')
#         start = time.time()
        
#         img_data = base64.b64decode(data['image'].split(',')[1])
#         nparr = np.frombuffer(img_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         analyzer = get_face_app()
#         faces = analyzer.get(img)
        
#         if len(faces) == 0:
#             return jsonify({'success': False, 'message': 'No face detected'}), 400
        
#         descriptor = faces[0].embedding.tolist()
#         processing_time = int((time.time() - start) * 1000)
        
#         print(f'✅ Extracted in {processing_time}ms')
        
#         return jsonify({
#             'success': True,
#             'descriptor': descriptor,
#             'confidence': 95,
#             'processingTime': processing_time
#         })
        
#     except Exception as e:
#         print(f'❌ Error: {str(e)}')
#         return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/face/extract', methods=['POST'])
def extract_face():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Image required'}), 400
        
        print('📸 Extracting face...')
        start = time.time()
        
        # Decode image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'message': 'Failed to decode image'}), 400
        
        # Log image details
        height, width = img.shape[:2]
        print(f'📐 Image size: {width}x{height}')
        
        # Try different detection sizes
        analyzer = get_face_app()
        
        # Try with multiple detection sizes for better results
        detection_sizes = [(640, 640), (320, 320), (160, 160)]
        faces = []
        
        for det_size in detection_sizes:
            print(f'🔍 Trying detection size: {det_size}')
            analyzer.prepare(ctx_id=0, det_size=det_size)
            faces = analyzer.get(img)
            
            if len(faces) > 0:
                print(f'✅ Found {len(faces)} face(s) with size {det_size}')
                break
        
        if len(faces) == 0:
            # Save failed image for debugging
            debug_path = f'/tmp/failed_{int(time.time())}.jpg'
            cv2.imwrite(debug_path, img)
            print(f'❌ No face detected. Image saved to {debug_path}')
            
            return jsonify({
                'success': False, 
                'message': 'No face detected. Please ensure face is clearly visible and well-lit.',
                'debug': {
                    'imageSize': f'{width}x{height}',
                    'brightness': int(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
                }
            }), 400
        
        # Use the best (first) face
        face = faces[0]
        descriptor = face.embedding.tolist()
        processing_time = int((time.time() - start) * 1000)
        
        # Calculate face quality score
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_area_ratio = (face_width * face_height) / (width * height)
        
        print(f'✅ Face extracted in {processing_time}ms')
        print(f'📊 Face area: {face_area_ratio*100:.1f}% of image')
        
        return jsonify({
            'success': True,
            'descriptor': descriptor,
            'confidence': 95,
            'processingTime': processing_time,
            'debug': {
                'faceCount': len(faces),
                'faceArea': f'{face_area_ratio*100:.1f}%',
                'imageSize': f'{width}x{height}'
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
        start = time.time()
        
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