import cv2
import os
from detect_text import detect_text_regions
from recognize_text import recognize_text_from_image

EAST_MODEL = "C:\\Users\\Lenovo\\OneDrive\\Documents\\Semester 4\\DIP\\Project\\STR\\models\\frozen_east_text_detection.pb"
IMAGE_PATH = "C:\\Users\\Lenovo\\OneDrive\\Documents\\Semester 4\\DIP\\Project\\STR\\images\\test.jpg"
OUTPUT_PATH = "C:\\Users\\Lenovo\\OneDrive\\Documents\\Semester 4\\DIP\\Project\\STR\\outputs\\result.jpg"

if not os.path.exists(EAST_MODEL):
    raise FileNotFoundError(f"EAST model not found at {EAST_MODEL}")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

image = cv2.imread(IMAGE_PATH)
orig, boxes = detect_text_regions(image, EAST_MODEL)

print(f"[INFO] Found {len(boxes)} text region(s)")

for (startX, startY, endX, endY) in boxes:
    roi = orig[startY:endY, startX:endX]
    if roi.size == 0:
        continue

    text = recognize_text_from_image(roi)
    print(f"[DETECTED] {text}")

    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(orig, text, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_PATH, orig)
print(f"[INFO] Result saved at {OUTPUT_PATH}")

# cv2.imshow("Text Detection", orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
