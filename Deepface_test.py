from deepface import DeepFace
import cv2

# Load image
img = cv2.imread("test.jpg")

# Analyze emotions
result = DeepFace.analyze(img, actions=['emotion'])

print(result)
