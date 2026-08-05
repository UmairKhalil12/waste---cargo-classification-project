import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models

# Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier
model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=5
)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# loss function
loss_function = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(model.fc.parameters() , lr=0.01)




    

    