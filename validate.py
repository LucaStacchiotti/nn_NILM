import torch
from MAE_ES import MAE,SAEd
# Definisce come fare il validate
def validate(model, criterion, val_dataload, device,inference):
  # set all network parameters to no-train mode
  model.eval()

  epoch_loss = 0.0
  num_MAE_mean=0.0
  num_SAE_mean=0.0


  with torch.no_grad():  # set the require_grad = false for validation

    for inputs, labels in val_dataload:
        # Send data to device: cuda or cpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        num_MAE=torch.zeros(1,5,1)
        num_SAE=num_MAE
        # accumulate loss
        epoch_loss += loss.item()
        if inference==True:
            num_MAE=num_MAE+MAE(outputs,labels)
            num_SAE=num_SAE+SAEd(outputs,labels,450)


  # compute mean loss and overall accuracy for the whole batch
  epoch_mean_loss = epoch_loss / len(val_dataload)
  if inference==True:  
    num_MAE_mean=num_MAE/len(val_dataload)
    num_SAE_mean=num_SAE/len(val_dataload)

  return epoch_mean_loss, num_MAE_mean, num_SAE_mean
