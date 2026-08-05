import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model_builder import device , model 
from data_setup import create_dataloaders , create_datasets , create_transforms


def predict( validation_data , number_of_predictions  , class_names , model,device):

    torch.manual_seed(42)
    max_number = len(validation_data)
    random_array = np.random.randint(0, max_number , size=number_of_predictions)

    model.to(device)
    model.eval()

    for i in random_array:
        image , label = validation_data[i] 
        image = image.to(device)
        image_model = image.unsqueeze(0)
        
        with torch.inference_mode():
            y_logits = model(image_model)
            y_preds = torch.argmax(y_logits, dim=1)

        pred_label = class_names[y_preds.item()]

        plt.imshow(image.cpu().permute(1,2,0))
        plt.title(f" Actual Label {class_names[label]} | Predicted label : {pred_label} ")
        plt.axis(False)
        plt.show()
        

    

model.load_state_dict(torch.load("cargo_resnet18.pth", map_location=device))
model.to(device)

train_transform , test_transform , val_transform = create_transforms()

train_data , test_data , val_data ,  class_names =  create_datasets(train_transform , test_transform , val_transform )

predict(val_data , number_of_predictions= 10, class_names = class_names , model=model , device=device)
            

