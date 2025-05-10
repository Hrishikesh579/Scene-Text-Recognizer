import cv2
import numpy as np

def decode_predictions(scores, geometry, conf_threshold=0.5):
    detections = []
    confidences = []

    height, width = scores.shape[2:4]
    for y in range(height):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(width):
            if scores_data[x] < conf_threshold:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)

            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
            endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            detections.append((startX, startY, endX, endY))
            confidences.append(scores_data[x])

    return detections, confidences

def detect_text_regions(image, east_model_path="models/frozen_east_text_detection.pb", width=320, height=320):
    orig = image.copy()
    (H, W) = image.shape[:2]

    newW, newH = width, height
    rW, rH = W / float(newW), H / float(newH)

    resized = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                                    (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net = cv2.dnn.readNet(east_model_path)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                        "feature_fusion/concat_3"])

    boxes, confidences = decode_predictions(scores, geometry)
    box_rects = [(x, y, ex - x, ey - y) for (x, y, ex, ey) in boxes]
    indices = cv2.dnn.NMSBoxes(box_rects, confidences, 0.5, 0.4)

    results = []

    if indices is not None and len(indices) > 0:
        # Normalize to array and flatten safely
        if isinstance(indices, tuple) or isinstance(indices, list):
            indices = np.array(indices)
        if len(indices.shape) == 2:
            indices = indices.flatten()

        for i in indices:
            startX, startY, endX, endY = boxes[i]
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            results.append((startX, startY, endX, endY))

    return orig, results
