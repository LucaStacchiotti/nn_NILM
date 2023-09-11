import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# Prende tutte le case in una lista e le concatena in un unico tensore
class REFITdatasetComp(Dataset):
        def __init__(self, path,filenames, len_win, step):
             self.win=len_win
             self.step=step
             self.data_len=[] #lista con lunghezze dataset pre-concatenati
             li=[]
             zero_row={'Aggregate':0, 'WM':0, 'DW':0, 'KE':0, 'FR':0, 'MI':0}
             for name in filenames:
                  df = pd.read_csv(path+name, index_col=None, header=0,usecols=[1,2,3,4,5,6])
                  while df.shape[0]%self.step!=0: # deve essere divisibile per step size, per evitare che un indice prenda 2 case 
                         df = df.append(zero_row, ignore_index=True)
                  self.data_len.append(df.shape[0])
                  li.append(df)
                  print('Loaded '+name)
             self.data = pd.concat(li, axis=0, ignore_index=True)       

             self.data=torch.tensor(self.data.values , dtype=torch.float32)
             mean_vect=torch.tensor([618,10,85,6,60,2])
             var_vect=torch.tensor([847,122,380,123,58,48])
             self.data=torch.div((torch.sub(self.data,mean_vect)),var_vect)
             self.data=torch.transpose(self.data,0,1)
        def __getitem__(self, index):                               
             x=self.data[0,self.step*index:self.step*index+self.win].unsqueeze(0)  
             y=self.data[1:6,self.step*index:self.step*index+self.win] #prende fino a :6-1
             return x,y
       
        # Usando le case in maniera concatenata, ma avendo trovato gli indici adatta per ogni casa singolo, bisogna shiftarli per la lunghezza della casa precedente
        def custom_len (self):
             l=[]
             for x in self.data_len:
                  l.append(x//self.step)
             return l
                                                 
        def __len__(self): #Siccome si usa un sampler custom, la lunghezza che ritorna è irrilevante
             l=1
             return l

# Analogo a quello per il REFIT, solo che i file sono .npy e non è presente la colonna dei tempi
class UKDALEdataset(Dataset):
        def __init__(self, path,filenames, len_win,step_size):
             self.win=len_win
             self.step=step_size
             self.data_len=[]
             first_bool=True

             for name in filenames:
                data=np.load(path+name)
                while data.shape[0]%self.step!=0:
                      data=np.vstack((data,[0,0,0,0,0,0]))
                self.data_len.append(data.shape[0])
                if first_bool==True:
                    data_tot=data
                    first_bool=False
                else:
                    data_tot=np.vstack((data_tot,data))
                print("Loaded "+name)

             self.data=torch.from_numpy(data_tot)
             mean_vect=torch.tensor([527,54,55,27,50,17])
             var_vect=torch.tensor([600,281,335,250,54,119])
             self.data=torch.div((torch.sub(self.data,mean_vect)),var_vect)
             self.data=torch.transpose(self.data,0,1).float()
        def __getitem__(self, index):
             x=self.data[0,self.step*index:self.step*index+self.win].unsqueeze(0)
             y=self.data[1:6,self.step*index:self.step*index+self.win]
             return x,y

        def custom_len (self): 
             l=[]
             for x in self.data_len:
                  l.append(x//self.step)
             return l
        def __len__(self): 
             l=1
             return l
