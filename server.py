import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('C:/Users/user/MDP/checklist_best.pt')

name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "1": 11,
        "2": 12,
        "3": 13,
        "4": 14,
        "5": 15,
        "6": 16,
        "7": 17,
        "8": 18,
        "9": 19,
        "A": 20,
        "B": 21,
        "C": 22,
        "D": 23,
        "E": 24,
        "F": 25,
        "G": 26,
        "H": 27,
        "S": 28,
        "T": 29,
        "U": 30,
        "V": 31,
        "W": 32,
        "X": 33,
        "Y": 34,
        "Z": 35,
        "Up": 36,
        "Down": 37,
        "Right": 38,
        "Left": 39,
        "Up Arrow": 36,
        "Down Arrow": 37,
        "Right Arrow": 38,
        "Left Arrow": 39,
        "Stop": 40
    }

custom_labels = {
    '11': '1',
    '12' : '2',
    '13': '3',
    '14': '4',
    '15' : '5',
    '16': '6',
    '17': '7',
    '18' : '8',
    '19': '9',
    '20': 'A',
    '21' : 'B',
    '22': 'C',
    '23': 'D',
    '24' : 'E',
    '25': 'F',
    '26': 'G',
    '27' : 'H',
    '28': 'S',
    '29': 'T',
    '30' : 'U',
    '31': 'V',
    '32' : 'W',
    '33': 'X',
    '34' : 'Y',
    '35': 'Z',
    '36': 'Up',
    '37' : 'Down',
    '38': 'Left',
    '39' : 'Right',
    '40': 'Stop',

    # Continue for other classes
}

@app.route('/status', methods=['GET'])
def status():
    """
    This is a health check endpoint to check if the server is running
    :return: a json object with a key "result" and value "ok"
    """
    return jsonify({"result": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the file as an image
        img = Image.open(file.stream)
        
        # Convert the image to a NumPy array
        img = np.array(img)
        
        # Convert RGB to BGR format (YOLO expects BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Run YOLOv8 inference
        results = model(img)  # This returns a list of Results objects

        result = results[0]   # Access the first Results object
        
        # Extract bounding boxes and labels from the results
        boxes = result.boxes  # Bounding boxes
        names = result.names  # Class names
        
        # Prepare predictions in a list of dictionaries
        predictions_short = []
        for box in boxes:
            class_name = names[int(box.cls[0].item())]
            boxHt = box.xyxy[0][3].item() - box.xyxy[0][1].item()
            boxWt = box.xyxy[0][2].item() -  box.xyxy[0][0].item()
            boxArea = boxHt * boxWt
            predictions_short.append({
                "x1": box.xyxy[0][0].item(),
                "y1": box.xyxy[0][1].item(),
                "x2": box.xyxy[0][2].item(),
                "y2": box.xyxy[0][3].item(),
                "confidence": box.conf[0].item(),
                "class_id":  name_to_id.get(class_name, 'NA'),
                "class_name": class_name,
                "box_area" : boxArea

            })
        if len(predictions_short) > 1:
            predictions = []
            for prediction in predictions_short:
                if prediction.get('x1') > 250 and prediction.get('x1') < 774:
                    predictions.append(prediction)
                    break
            if not predictions:
                predictions_short.sort(key=lambda x: x.get('box_area'), reverse=True)
                predictions = predictions[-1]
        elif len(predictions_short) == 0:
            predictions = []
            predictions.append({
                "x1": 0,
                "y1": 0,
                "x2": 0,
                "y2": 0,
                "confidence": 0,
                "class_id":  'NA',
                "class_name": 'NA',
                "box_area" : 0
            })
        else:
            predictions = [predictions_short[0]]
        
        return jsonify(predictions)
    #box.cls[0].item()
#name_to_id.get(class_name, 'NA')
#custom_labels.get(class_name, 'NA')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
