# nn_NILM
Codice rete neurale in Pytorch per progetto esame

I file "main.py", "EDNNet.py", "train.py", "validate.py", "MAE_ES.py", "saveModel.py" e "Dataset.py" sono in Pytorch e servono per definere rete e dataset, fare il training e validation e salva il migliore modello ottenuto.

Per scegliere quale dataset usare (REFIT o U.K.-DALE) bisogna commentare le righe appropiate sul main.py, oltre a cambiare i path delle cartelle. I file dei dataset sono .csv e .npy che sono stati ottenuti da quelli originali, dove si sono tolti gli eletrodomestici non usati e, nel caso dello U.K-DALE, sono stati riallineati gli elettrodomestici con la pot. aggregata.
Per ogni casa dei dataset, si sono scelte solo alcune sequenze, i cui indici sono i file all'interno della cartella "idx_subset" che sono stati trovati con "check_data_off_frame.ipynb" e "check_data_off_frame.ipynb"

Per fare inferenza, si usa "inference.ipynb"
