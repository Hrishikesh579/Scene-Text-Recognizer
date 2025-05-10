from flask import Flask, render_template, request
import os
import cv2
import uuid
from STR.detect_text import detect_text_regions
from STR.recognize_text import recognize_text_from_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'STR/models/frozen_east_text_detection.pb'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    recognized_texts = []
    input_image = None
    output_image = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = str(uuid.uuid4()) + '.jpg'
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            result_path = os.path.join(RESULT_FOLDER, 'result_' + filename)
            file.save(upload_path)

            image = cv2.imread(upload_path)
            orig, boxes = detect_text_regions(image, MODEL_PATH)

            for (startX, startY, endX, endY) in boxes:
                roi = orig[startY:endY, startX:endX]
                if roi.size == 0:
                    continue
                text = recognize_text_from_image(roi)
                recognized_texts.append(text)
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(orig, text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite(result_path, orig)
            input_image = upload_path
            output_image = result_path

    return render_template('index.html',
                            input_image=input_image,
                            output_image=output_image,
                            texts=recognized_texts)

if __name__ == '__main__':
    app.run(debug=True)
