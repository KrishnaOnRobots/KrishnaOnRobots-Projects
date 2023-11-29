import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',26: 'nothing', 27: 'space',28: 'del'}

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    transforms.Normalize((0.1307,), (0.3081,))
])

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 29)

filename = 'sign_lang_model.pth'
state = torch.load(filename)
model.load_state_dict(state)

model.to(device)
model.eval()
# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the digits
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = frame.shape[:2]

    # Set crop dimensions
    # crop_height = crop_width = min(height, width)

    # Calculate coordinates of the center of the image
    center_y = height // 2
    center_x = width // 2

    # Calculate coordinates of top-left corner of crop
    crop_y = center_y - 150
    crop_x = center_x - 150

    # Crop the center of the image
    cropped_image = frame[crop_y:crop_y+150, crop_x:crop_x+150]
    cv2.imshow('frame_1', cropped_image)
    cv2.rectangle(frame, (crop_x, crop_y), (crop_x + 150, crop_y + 150), (0, 255, 0), 2)

    img = torch.from_numpy(cropped_image)
    img = img.permute(2, 0, 1)
    img = img.to(device)
    img = img.to(torch.float32)
    img = img.unsqueeze(0)
    output = model(img)
    print(output)
    pred = torch.argmax(output, 1).item()

    cv2.putText(frame, str(dict[pred]), (crop_x, crop_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # print(f"Predicted: {dict[pred]}")

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
