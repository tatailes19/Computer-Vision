import torch

# Load the model architecture
from models.CDCNs import CDCNpp  # Replace with your model class import

# Instantiate the model
model = CDCNpp()

# Load the trained weights
checkpoint = torch.load('experiments/output/CDCNpp_nuaa.pth',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set the model to evaluation mode

from ultralytics import YOLO
import cv2
import tensorflow as tf 
import keras
from keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np
import os

preprocess = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL Image
    transforms.Resize(cfg['model']['input_size']),  # Resize
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma']),  # Normalize
])

face_model = YOLO("models/yolov8n-face.pt")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    

    # Display the live video stream
    cv2.imshow("Camera", frame)

    # Capture an image when 'c' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        
        results = face_model(frame)
        
        input_tensor = preprocess(frame).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        
        predicted_classes = torch.argmax(outputs[0], dim=1)
        
        yhat = torch.mode(predicted_classes).values.item()
        
        if yhat == 1:
            label = 'Real'
        else:
            label = 'Fake'


        for result in results[0].boxes:
            top_left_x = int(result.xyxy.tolist()[0][0])
            top_left_y = int(result.xyxy.tolist()[0][1])
            bottom_right_x = int(result.xyxy.tolist()[0][2])
            bottom_right_y = int(result.xyxy.tolist()[0][3])
        
        
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (255, 0, 0), 2)
        cv2.putText(frame, label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv2.imshow("Captured Image", frame)

        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()