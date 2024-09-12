from flask import Flask, jsonify
from picamera import PiCamera
import os
import time

app = Flask(__name__)

# Initialize the camera
camera = PiCamera()

# Folder to save captured images
image_folder = "/home/pi/captured_images"
server_url = 'http://127.0.0.1:5001/predict'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

@app.route('/take_picture', methods=['POST'])
def take_picture():
    try:
        # Generate filename with timestamp
        filename = f"{int(time.time())}_image.jpg"
        image_path = os.path.join(image_folder, filename)

        # Start the camera and take a picture
        camera.start_preview()
        time.sleep(2)  # Warm-up time
        camera.capture(image_path)
        camera.stop_preview()

        with open(image_path, 'rb') as file:
            files = {'file': (filename, file)}
            response = requests.post(server_url, files=files)

        return response

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Make sure this port matches the URL used in the Flask server