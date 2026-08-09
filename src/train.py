import torch.nn as nn
import torch
from model_builder import device , model , optimizer , loss_function
from tqdm.auto import tqdm
from data_setup import create_dataloaders
from utils import classification_report_function , loss_curve , accuracy_curve , heat_map_confusion_matrix
from predict import evaluate_model



def val_step(model,val_dataloader,loss_fn,device):
  model.eval()

  val_loss = 0
  val_acc = 0

  with torch.inference_mode():
    for X , y in val_dataloader:
      X = X.to(device)
      y = y.to(device)

      val_logit = model(X)
          
      loss = loss_fn(val_logit , y)
      val_loss += loss.item()

      val_pred = torch.argmax(val_logit , dim=1)

      val_acc += (val_pred == y).float().mean().item()

  val_loss /= len(val_dataloader)
  val_acc /= len(val_dataloader)

  return val_loss , val_acc




def train_step(
    model:nn.Module,
    dataloader : torch.utils.data.DataLoader,
    loss_fn : nn.Module,
    optimizer : torch.optim.Optimizer,
    device = device
):
  # Put the model in train mode
  model.train()

  model.to(device)

  # Setup train and loss accuracy values
  train_loss , train_acc = 0 , 0

  # Loop through dataloader
  for batch , (X,y) in enumerate(dataloader):
    # Send data to target device
    X , y = X.to(device) , y.to(device)

    # 1. Forward pass
    y_pred = model(X)

    # 2. Calculate the loss
    loss = loss_fn(y_pred , y)
    train_loss += loss.item()

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Calculate the accuracy
    y_pred_class = torch.argmax(y_pred, dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss , train_acc

def test_step(
    model:nn.Module,
    dataloader : torch.utils.data.DataLoader,
    loss_fn : nn.Module,
    device = device
):
  model.eval()

  # Setup test loss and test acc values
  test_loss , test_acc = 0 , 0
  with torch.inference_mode():
    for batch , (X,y) in enumerate(dataloader):
      # Send data to target device
      X , y = X.to(device) , y.to(device)

      # 1. Forward pass
      test_pred_logits = model(X)

      # 2. Calculate the loss
      loss = loss_fn(test_pred_logits , y)
      test_loss += loss.item()

      # Calculate the accuracy
      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)

  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss , test_acc


def train(model:torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader : torch.utils.data.DataLoader,
          val_dataloader : torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer,
          loss_fn : torch.nn.Module = nn.CrossEntropyLoss(),
          epochs : int = 5,
          device =  device):
  # 2. Create  result dictionary
  results = {"train_loss" : [],
             "train_acc":[],
             "test_loss": [],
             "test_acc":[],
             "val_loss" : [],
             "val_acc" : []
             }
  # 3. Loop through training and testing steps for a number of epcohs

  best_acc_score = 0
  for epoch in tqdm(range(epochs)):
    train_loss , train_acc = train_step(model=model,
                                        dataloader = train_dataloader,
                                        loss_fn = loss_fn,
                                        optimizer = optimizer,
                                        device = device)

    test_loss , test_acc = test_step(model=model,
                                      dataloader = test_dataloader,
                                      loss_fn = loss_fn,
                                      device = device)

    val_loss,  val_acc = val_step(model=model , 
                                  val_dataloader=val_dataloader ,
                                    loss_fn=loss_fn,
                                    device=device)


    # 4. Print out what's happening
    print(f"Epoch : {epoch} | Train loss : {train_loss : .4f} | Train acc :{train_acc: .4f} | Test Loss : {test_loss : .4f} | Test acc : {test_acc : .4f} | Validation Loss : {val_loss : .4f} | Validation acc : {val_acc : .4f}")

    # 5. update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)

    if(best_acc_score < results["val_acc"][epoch]):
      best_acc_score = results["val_acc"][epoch]
      torch.save(model.state_dict(),"cargo_resnet18.pth")

    

  # 6. Return the filled results at the end of epochs
  return results


if __name__ == "__main__":

    # =================================================== TRAIN =============================== //
    train_dataloaders, test_dataloaders, val_dataloader, class_names = create_dataloaders()

    # results_training = train(
    #     model=model,
    #     train_dataloader=train_dataloaders,
    #     test_dataloader=test_dataloaders,
    #     val_dataloader=val_dataloader,
    #     optimizer=optimizer,
    #     loss_fn=loss_function,
    #     epochs=5,
    #     device=device
    # )

    # print(results_training , 'results for training')
    # torch.save(results_training, "training_results.pth")
  
    # -------------------- SAVING MODEL ------------------------------- //
    # already saving the model's best epoch
    model.load_state_dict(
      torch.load("cargo_resnet18.pth", map_location=device)
    )

    model.to(device)


    # ============== EVALUATION -------------------------- //

    # classification_report,  confusion_matrix = classification_report_function(model=model ,
    #                                 test_dataloader=test_dataloaders,
    #                                 class_names=class_names,
    #                                 device=device)

    # torch.save(classification_report , "classification_report.pth")
    # torch.save(confusion_matrix , "confusion_matrix.pth")



    # ------------------ EVALUATE MODEL ---------------- //
    train_dataloader , test_dataloader , val_dataloader ,  class_names = create_dataloaders()
    report , matrix = evaluate_model(model , test_dataloader=test_dataloader,device=device,class_names=class_names)
    print(report)
    print(matrix)

    results_training = torch.load("training_results.pth", map_location=device)
    loss_curve(results=results_training)
    accuracy_curve(results=results_training)

    heat_map_confusion_matrix(confusion_report=matrix,class_names=class_names)





    

   

