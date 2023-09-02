import cv2
import pytesseract

def detect_text_tesseract(image_path):
    # Read the image from file
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(gray_image)

    return text.strip()

# Example usage:
if __name__ == "__main__":
    image_path = "/Users/apple/Downloads/Frames/frame_00042.jpg"  # Replace with the path to your image
    recognized_text = detect_text_tesseract(image_path)
    print("Recognized Text:", recognized_text)
