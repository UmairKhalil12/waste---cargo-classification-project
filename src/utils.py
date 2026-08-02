import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

loss_curve(results=results_training)
accuracy_curve(results=results_training)