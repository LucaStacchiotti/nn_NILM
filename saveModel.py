import torch
# Salva il modello (parametri, gradienti, ecc...)


def save_model(epochs, model, optimizer):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'best_param.pth')
    print(f"Model saved")
