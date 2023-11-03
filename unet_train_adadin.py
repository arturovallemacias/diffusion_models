
import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim 
from model import UNET

#from utils import ( 
    #load_checkpoint,
    #save_checkpoint, 
    #get_loaders,
    #check_accuracy,
    #save_predictions_as_imgs, 
    
#)

LEARNING_RATE = 1e-4 
DEVICE = "cuda" if torch.cuda.is_availabe() else "cpu"
BATCH_SIZE = 16 
NUM_EPOCHS = 3
NUM_WORKERS = 2 
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 240 
PIN_MEMORY = True 
LOAD_MODEL = False 
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images"+
VAL_MASK_DIR = "data/val_masks"

def train_fn(loader, model, optimizer, loss_fn, scaler): 
    loop = tqdm(loader) 
    for batch_idx, (data, targets) in enumerate(loop):
        
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        with torch.cuda.amp.autocast(): 
            predictions = model(data)
            loss = loss_fn(predictions, targets)
          

def main(): 
    pass

if __name__ == "__main__": 
    main()
