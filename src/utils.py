import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report , confusion_matrix
from model_builder import device

from data_setup import create_dataloaders

import torch
results_training = torch.load("training_results.pth")

#from train import results_training

def accuracy_curve(results):
    acc_table = pd.DataFrame(results)
    length_of_dataframe = acc_table.shape[0]
    acc_table["epochs"] = np.arange(1,length_of_dataframe+1)
    plt.title("Accuracy Graph")
    sns.lineplot(data=acc_table , x="epochs" , y="train_acc" , label="train acc" )
    sns.lineplot(data=acc_table , x="epochs" , y="test_acc" , label="test acc")
    sns.lineplot(data=acc_table , x="epochs" , y="val_acc" , label="val acc")
    plt.legend()
    plt.show()

def loss_curve(results):
    loss_table = pd.DataFrame(results)
    length_of_dataframe = loss_table.shape[0]
    loss_table["epochs"] = np.arange(1,length_of_dataframe+1)
    plt.title("Loss Graph")
    sns.lineplot(data=loss_table , x="epochs" , y="train_loss",label="train loss")
    sns.lineplot(data=loss_table , x="epochs" , y="test_loss" , label="test loss")
    sns.lineplot(data=loss_table , x="epochs" , y="val_loss" , label="val loss")
    plt.legend()
    plt.show()

def classification_report_function(model,test_dataloader,class_names,device):
  model.to(device)
  model.eval()
  y_true = []
  y_pred = []

  with torch.inference_mode():
    for X , y in test_dataloader:
      X = X.to(device)
      y = y.to(device)

      y_logit = model(X)
      predictions = torch.argmax(y_logit , dim=1)
      
      y_true.extend(y.cpu().numpy())
      y_pred.extend(predictions.cpu().numpy())

    
    final_report = classification_report(y_true, y_pred, target_names=class_names)
    print(f" ----- Classification report -----------  \n" )
    print(final_report)

    confusion_report = confusion_matrix(y_true,y_pred)
    print(f"-------- Confusion report --------------------- \n")
    print(confusion_report)

  return final_report , confusion_report

def heat_map_confusion_matrix(confusion_report , class_names):
   sns.heatmap(confusion_report ,
                annot=True ,
                xticklabels=class_names,
                yticklabels=class_names,
                fmt="d")
   
   plt.show()

#loss_curve(results=results_training)
#accuracy_curve(results=results_training)

#train_dataloaders, test_dataloaders, val_dataloader, class_names = create_dataloaders()
#confusion_matrix_loaded = torch.load("confusion_matrix.pth",map_location=device, weights_only=False)
#heat_map_confusion_matrix(confusion_report=confusion_matrix_loaded , class_names=class_names)