import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model_builder import device , model 
from data_setup import  create_datasets , create_transforms , create_dataloaders
from sklearn.metrics import classification_report , confusion_matrix


def predict( validation_data , number_of_predictions  , class_names , model,device):
    #torch.manual_seed(42)
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
        probabilites = torch.softmax(y_logits , dim=1)
        confidence = probabilites.max(dim=1).values.item()
        plt.title(f" Actual Label : {class_names[label]} | Predicted label : {pred_label} | Confidnece : {confidence:.2%} " , fontsize=8)
        plt.imshow(image.cpu().permute(1,2,0))
        plt.axis(False)
        plt.show()


def evaluate_model(model,test_dataloader,device,class_names):
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.inference_mode():
        for X , y in test_dataloader:
            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)
            predictions = torch.argmax(y_logits,dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

        class_report = classification_report(y_true=y_true , y_pred=y_pred , target_names=class_names)

        confuse_matrix = confusion_matrix(y_true , y_pred)

        return class_report, confuse_matrix
    




model.load_state_dict(torch.load("cargo_resnet18.pth", map_location=device))
model.to(device)

# -------------- PREDICTIONS ================= // 

#train_transform , test_transform , val_transform = create_transforms()
#train_data , test_data , val_data ,  class_names =  create_datasets(train_transform , test_transform , val_transform )
#predict(val_data , number_of_predictions= 10, class_names = class_names , model=model , device=device)




            

