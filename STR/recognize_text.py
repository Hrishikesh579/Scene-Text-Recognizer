import cv2
import pytesseract

# Set Tesseract path if using Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def recognize_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = "--psm 6"  # Assume a block of text
    text = pytesseract.image_to_string(gray, config=config)
    return text.strip()
