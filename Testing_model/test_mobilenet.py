import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

"""FER2013"""

class_names=['angry','Disgust','Fear','Happy','Sad','Surprised','Neutral']

# Applying transform function
transform=transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# FER2013 dataset- Loading the FER2013 dataset
train_dir= '/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/Face_Emotion_Analysis/DATA/FER_train'
test_dir='/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/Face_Emotion_Analysis/DATA/FER_test'

train_fer=ImageFolder(root=train_dir,transform=transform)
test_fer=ImageFolder(root=test_dir,transform=transform)
test_fer, valid_fer = train_test_split(test_fer, test_size=0.02, random_state=42)

trainfer_loader= DataLoader(train_fer, batch_size=64, shuffle=True)
testfer_loader= DataLoader(test_fer, batch_size=64, shuffle=True)
validfer_loader=DataLoader(valid_fer,batch_size=64,shuffle=True)


# FER2013Plus dataset- Loading the EMOTIC dataset
train_dir2= '/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/Face_Emotion_Analysis/fer2013plus/fer2013/train'
test_dir2='/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/Face_Emotion_Analysis/fer2013plus/fer2013/test'

train_fer2=ImageFolder(root=train_dir2,transform=transform)
test_fer2=ImageFolder(root=test_dir2,transform=transform)
test_fer2, valid_fer2 = train_test_split(test_fer, test_size=0.02, random_state=42)

train_loader= DataLoader(train_fer2, batch_size=64, shuffle=True)
test_loader= DataLoader(test_fer2, batch_size=64, shuffle=True)
valid_loader=DataLoader(valid_fer2,batch_size=64,shuffle=True)


model_mobilenet=models.mobilenet_v2(pretrained=True)



num_features = model_mobilenet.classifier[1].in_features
num_classes=len(train_fer.classes)
print(num_classes)

custom_classifier=nn.Sequential(
    nn.Linear(num_features,512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512,num_classes)
)

model_mobilenet.classifier=custom_classifier

model_mobilenet.to(device)

# model_mobilenet.classifier.load_state_dict(torch.load('model_mobilenet.pth'))



criterion=nn.CrossEntropyLoss().to(device)
optimizer=optim.Adam(model_mobilenet.parameters(),lr=0.001)

state_dict = torch.load('/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/Face_Emotion_Analysis/checkpoint_acc97.4816.pth')
# model_mobilenet.load_state_dict(state_dict)
model_mobilenet.load_state_dict(state_dict['state_dict'])

# model_mobilenet.eval()

# Initialize variables for calculating accuracy
correct = 0
total = 0

# Iterating through FER2013 test dataset
with torch.no_grad():
    correct=0
    total=0
    for images,labels in testfer_loader:
        images,labels =images.to(device),labels.to(device)
        
		#Forward pass
        outputs=model_mobilenet(images)
        
        #Get predicted labels
        _,predicted=torch.max(outputs.data,1)
        
		# Update total and correct predictions
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        
    # Calculate accuracy  
    accuracy=100*correct/total
    print(f'Test Accuracy for FER2013: {accuracy:.2f}%')
    


# Iterating through EMOTIC test dataset
with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_loader:
        images,labels =images.to(device),labels.to(device)
        
		#Forward pass
        outputs=model_mobilenet(images)
        
        #Get predicted labels
        _,predicted=torch.max(outputs.data,1)
        
		# Update total and correct predictions
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        
    # Calculate accuracy  
    accuracy=100*correct/total
    print(f'Test Accuracy for FER2013 Plus: {accuracy:.2f}%')
    


