import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models
from data_setup import create_dataloaders

train_dataloader , test_dataloader , val_dataloader ,  class_names = create_dataloaders()

def create_model(num_classes):
    # Load pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final classifier
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=len(class_names)
    )

    return model

model = create_model(num_classes=len(class_names))

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# loss function
loss_function = nn.CrossEntropyLoss()

# optimizer
#optimizer = optim.SGD(model.fc.parameters() , lr=0.001)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)




    

    