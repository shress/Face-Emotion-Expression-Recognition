import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np 
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)

"""FER2013"""

class_names=['angry','Disgust','Fear','Happy','Sad','Surprised','Neutral']



# Applying transform function
transform=transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# FER2013 dataset
train_dir= '/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/Face_Emotion_Analysis/DATA/FER_train'
test_dir='/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/Face_Emotion_Analysis/DATA/FER_test'

train_fer=ImageFolder(root=train_dir,transform=transform)
test_fer=ImageFolder(root=test_dir,transform=transform)
test_fer, valid_fer = train_test_split(test_fer, test_size=0.02, random_state=42)

trainfer_loader= DataLoader(train_fer, batch_size=64, shuffle=True)
testfer_loader= DataLoader(test_fer, batch_size=64, shuffle=True)
validfer_loader=DataLoader(valid_fer,batch_size=64,shuffle=True)

#defining the pretrained models 

#MobileNetV2 - take (224,224,3) 

num_classes=len(train_fer.classes)

model_mobilenet=models.mobilenet_v2(pretrained=True)



num_features = model_mobilenet.classifier[1].in_features

custom_classifier=nn.Sequential(
    nn.Linear(num_features,512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512,num_classes)
)

model_mobilenet.classifier=custom_classifier

model_mobilenet.to(device)


modelSavedir = os.path.join(os.getcwd(),'model_mobilenet3')
os.makedirs(modelSavedir, exist_ok=True)

model_mobilenet.to(device)

criterion=nn.CrossEntropyLoss().to(device)
optimizer=optim.SGD(model_mobilenet.parameters(),lr=0.0001)


num_epochs=250

for epoch in range(num_epochs):
    running_loss=0
    correct=0
    total=0
    for i, data in enumerate(trainfer_loader, 0):
        inputs,labels=data
        inputs,labels= inputs.to(device),labels.to(device)
        
        optimizer.zero_grad()
        
        outputs=model_mobilenet(inputs)
        
        loss =criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss=loss.item()
        
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        
        accuracy=100*correct/total
        
        # print(f'Epochs: [{epoch+1}/{num_epochs}], iter[{i}/{len(trainfer_loader)}]  Loss: {loss.item():.4f}  Accuracy: {accuracy}')
    
    print(f'Epochs: {epoch+1}/{num_epochs}, item[{i}/{len(trainfer_loader)}]  Loss: {running_loss/len(trainfer_loader):.4f}  Accuracy: {accuracy}')
    
    if epoch==0:
        best_acc=0
        
    checkpoint={'epoch':epoch+1, 'state_dict': model_mobilenet.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint,os.path.join(modelSavedir,f'checkpoint_mobilenet.pth' )) 
    
    if accuracy>best_acc:
        torch.save(checkpoint, os.path.join(modelSavedir, f'checkpoint_acc{accuracy:.4f}.pth'))
  
    best_acc=accuracy

print("Training Completed!!")

state_dict = torch.load('/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/SHREYA/Face_Emotion_Analysis/model_mobilenet.pth')
# model_mobilenet.load_state_dict(state_dict)
model_mobilenet.load_state_dict(state_dict['state_dict'])


model_mobilenet.eval()

with torch.no_grad():
    correct=0
    total=0
    for images,labels in testfer_loader:
        images,labels =images.to(device),labels.to(device)
        outputs=model_mobilenet(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        
    accuracy=100*correct/total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
