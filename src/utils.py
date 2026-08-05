import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report , confusion_matrix

import torch
results_training = torch.load("training_results.pth")

#from train import results_training

def accuracy_curve(results):

    acc_table = pd.DataFrame(results)
    #print(acc_table.head())

    length_of_dataframe = acc_table.shape[0]

    acc_table["epochs"] = np.arange(1,length_of_dataframe+1)

    #print(acc_table.head())

    plt.title("Accuracy Graph")
    sns.lineplot(data=acc_table , x="epochs" , y="train_acc" , label="train acc" )
    sns.lineplot(data=acc_table , x="epochs" , y="test_acc" , label="test acc")
    sns.lineplot(data=acc_table , x="epochs" , y="val_acc" , label="val acc")
    plt.legend()

    plt.show()

def loss_curve(results):
    loss_table = pd.DataFrame(results)
    #print(loss_table.head())
    
    length_of_dataframe = loss_table.shape[0]

    loss_table["epochs"] = np.arange(1,length_of_dataframe+1)

    #print(loss_table.head())
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

#loss_curve(results=results_training)
#accuracy_curve(results=results_training)