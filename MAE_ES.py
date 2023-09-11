import torch
import numpy as np

# Definisce alcune metriche e strumenti 


#Calcola MAE
def MAE(ye,ygt): #Entra BATCHxAPPLxNT, esce 1xAPPLx1
  temp=ye-ygt
  temp= torch.abs(temp)
  MAE=torch.sum(temp,2)/600
  MAE=torch.sum(MAE,0)/256
  return MAE
  
#Calcola SAE, entra BATCHxAPPLxNT, esce 1xAPPLx1
def SAEd(ye,ygt,Td):
  ye=ye[:,:,0:Td]
  ygt=ygt[:,:,0:Td]
  temp_e=torch.sum(ye,2) #energia
  temp_gt=torch.sum(ygt,2)
  SAE=torch.rand(256,5)
  for i in range(5):
    SAE[:,i]=(torch.abs(temp_e[:,i]-temp_gt[:,i])/temp_gt[:,i])/Td
  SAE=torch.sum(SAE,0)/256
  return SAE
  
#Early Stopper
#Se la loss non migliora in "patience" epoche , ferma il training
class EarlyStopper:
   def __init__(self, patience=1, min_delta=0):
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.min_validation_loss = np.inf

   def early_stop(self, validation_loss):
       if validation_loss < self.min_validation_loss:
           self.min_validation_loss = validation_loss
           self.counter = 0
       elif validation_loss > (self.min_validation_loss + self.min_delta):
           self.counter += 1
           if self.counter >= self.patience:
               return True
       return False