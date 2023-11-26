import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

# Function to predict emotion
def predict_emotion(model, face):
    model.eval()
    with torch.no_grad():
        output = model(face)
    probabilities = F.softmax(output, dim=1)
    predicted_emotion_index = torch.argmax(probabilities).item()
    return predicted_emotion_index

# Load pre-trained MobileNetV2 model
model_mobilenet = models.mobilenet_v2(pretrained=True)

# Customize the classifier for your specific task
num_classes = 7
num_features = model_mobilenet.classifier[1].in_features
custom_classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
model_mobilenet.classifier = custom_classifier

# Load the pre-trained model's weights
state_dict = torch.load('C:/D drive/Bennett/VS/models/checkpoint_acc75.5278.pth', map_location=torch.device('cpu'))
model_mobilenet.load_state_dict(state_dict['state_dict'])
model_mobilenet.eval()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral','Sad', 'Surprised']

# Create face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create video capture object
cap = cv2.VideoCapture(0)

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

while cap.isOpened():
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        if w > 0 and h > 0:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face_tensor = transform(face).unsqueeze(0)
            emotion_prediction = predict_emotion(model_mobilenet, face_tensor)
            recognized_emotion = emotion_labels[emotion_prediction]

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(img, recognized_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face detection:", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
