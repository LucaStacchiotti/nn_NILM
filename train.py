import torch

# Definisce come fare il train
def train(model, optimizer, criterion, train_dataload, device):
  # set all network parameters to train mode
  model.train()
  epoch_loss = 0.0
  #for i, data in enumerate(train_dataload):
  for inputs, labels in train_dataload:
      # Send data to device: cuda or cpu
      inputs = inputs.to(device)
      labels = labels.to(device)

      # Set the parameter gradients to zero
      optimizer.zero_grad()
      #for param in model.parameters():
      #        param.grad = None

      # forward pass
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      # backward propagation
      loss.backward()

      # gradient clipping
      torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5,norm_type=2.0)

      # optimization (update weights)
      optimizer.step()

      # accumulate loss
      epoch_loss += loss.item()

  # compute mean loss across batch
  epoch_mean_loss = epoch_loss / len(train_dataload)

  return epoch_mean_loss
