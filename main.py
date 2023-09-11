import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import time
import numpy as np
import json
from EDNNet import ConvTasNet
from Dataset import UKDALEdataset, REFITdatasetComp
from train import train
from validate import validate
from MAE_ES import EarlyStopper, MAE
from torch.utils.data.sampler import SubsetRandomSampler
from saveModel import save_model


#### Parametri ####
num_batch=256
l_rate=0.001;
epoche=150;
num_appl=5;
length_win=600;
filepath='~/code/refit/'
#filepath='../code/ukdale/'
n_workers=0
list_h_idxs=['1','2','3','6','7','8','9','10','12','15','17','18','19','20']
#list_h_idxs=['1','5']
list_h_subset=['h'+x for x in list_h_idxs]
#list_h_subset=['h'+x+'_ukdale' for x in list_h_idxs]
list_h_refit=['H'+x+'_t.csv' for x in list_h_idxs]
list_h_ukdale=['H'+x+'_ukd.npy' for x in list_h_idxs]
    
#### Utilizza GPU e implementa NN ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ###########
#print(device)
model= ConvTasNet(num_sources=num_appl,enc_kernel_size=10,enc_num_feats=512,msk_num_feats=256,msk_num_hidden_feats=512,msk_kernel_size=7,msk_num_layers=4,msk_num_stacks=4)
model=model.to(device)
#print(model)    
#### Stampa numero parametri allenabili ####
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"The network has {params} trainable parameters")
    
    
#### Carica Dataset ####
data=REFITdatasetComp(filepath,list_h_refit,length_win,100)
#data=UKDALEdataset(filepath,list_h_ukdale,length_win,75)
val_data=REFITdatasetComp(filepath,['H5_t.csv'],length_win,100)
#val_data=UKDALEdataset(filepath,['H2_ukd.npy'],length_win,75)
    
### Combina gli indici ricavati in una unica sequenza ###
sum_len=0
c=0
total_idx=[]
temp_idx=[]
for house in list_h_subset:
    with open("idx_subset/"+house+"_idx_subset.json", 'r') as f:
        temp_idx=(json.load(f))
    temp_idx=[x+sum_len for x in temp_idx]
    #print(len(temp_idx))
    total_idx=list(set(total_idx+temp_idx))
    #print(len(total_idx))
    print(house)
    sum_len=sum_len+data.custom_len()[c]
    c=c+1    
# Deve essere divisibile per num_batch; cancella l'ultimo
while len(total_idx)%num_batch!=0:
    del total_idx[-1]

# Indici per validation
#with open("idx_subset/h2_ukdale_idx_subset_val.json", 'r') as f:
with open("idx_subset/h5_idx_subset_val.json", 'r') as f:
     val_idx=(json.load(f))
while len(val_idx)%num_batch!=0:
    del val_idx[-1]


#print(data.custom_len())
train_dataload = torch.utils.data.DataLoader(data, batch_size=num_batch, num_workers=n_workers,sampler=SubsetRandomSampler(total_idx),pin_memory=True)
val_dataload=torch.utils.data.DataLoader(val_data, batch_size=num_batch, num_workers=n_workers, pin_memory=True,sampler=SubsetRandomSampler(val_idx))
print(f'{len(train_dataload)*num_batch} frames for training ({len(train_dataload)} batches of {num_batch} images)')
   
total_idx=[]
temp_idx=[]
val_idx=[]

######################################################
############## TRAINING e VALIDATION #################
######################################################
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
val_loss_best=10000000

print(f'Start training with {device}')
   
train_loss_list = []
val_loss_list = []

# get timestamp
training_start_time = time.time()

# Se per 10 epoche la validation loss non migliora, ferma il training
early_stopper = EarlyStopper(patience=10, min_delta=0)

# Dimezza lr se non migliora la validate loss per 4 epoche
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,patience=4)

# Ottimizzatore hardware
torch.backends.cudnn.benchmark = True

for epoch in range(epoche):
     epoch_time = time.time()

     epoch_train_loss = train(model, optimizer, criterion, train_dataload,device)
     print(f'Epoch {epoch + 1}/{epoche} - Train Loss: {epoch_train_loss:.4} -' , end='')
     # track the train loss and accuracy
     train_loss_list.append(epoch_train_loss)

     # at the end of the epoch do a pass on the validation set
     epoch_val_loss,_,_ = validate(model, criterion, val_dataload,device,False)

     print(f' Val. Loss: {epoch_val_loss:.4} - Elapsed time: {(time.time() - epoch_time):.1f} s')

     if epoch_val_loss<val_loss_best:
         save_model(epoch,model,optimizer) # Salva parametri
         val_loss_best=epoch_val_loss

     val_loss_list.append(epoch_val_loss)

     #LROnPlateau
     scheduler.step(epoch_val_loss)

     # Early Stopper
     if early_stopper.early_stop(epoch_val_loss):
       print('\nEarly stopping: val. loss did not improve')
       break
print('-'*20)
print(f'Training finished, Total elapsed time {(time.time() - training_start_time)/60:.2f} min\n\n')
    
# Salva valore loss function per ogni epoca
with open("loss_list.json", 'w') as f:   
     json.dump(train_loss_list+val_loss_list, f, indent=2)

