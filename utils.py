import matplotlib.pyplot as plt
import numpy as np
import os, time
from datetime import datetime

from services import EarlyStopper

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hrs = int(elapsed_time / 3600)
    elapsed_mins = int((elapsed_time - elapsed_hrs * 3600) / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60 + elapsed_hrs * 3600))
    return elapsed_hrs,  elapsed_mins, elapsed_secs

def getModelName(config):
    now = str(datetime.now())
    date, time = now.split()[0], now.split()[1]
    date = date.split('-')
    date.reverse()
    date = '-'.join(date)
    time = time.replace(':', '-')[:8]
    
    model_name = f"Model_{config.epochs}_{date}_{time}"
    return model_name

def generatePlots(train_list, val_list, fig_path, plot_type = 'loss'):
    if plot_type == 'loss':
        if len(train_list) == 0 or len(val_list) == 0:
            print("List empty")
        else:
            min_val_loss = min(val_list)
            epoch_loss = val_list.index(min_val_loss)
            print(f"Optimal point : {epoch_loss} epoch with Val loss {min_val_loss}")
            plt.figure()
            plt.plot(range(len(train_list)), train_list, color='blue', label='Train Loss', linestyle='dashed')
            plt.plot(range(len(val_list)), val_list, color='green', label='Valid loss', linestyle='dashed')
            plt.plot(epoch_loss, min_val_loss, marker = 'v', color = 'red', label = 'Min Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Summary : Loss')
            plt.legend()
            plt.savefig(fig_path)
    elif plot_type == 'acc':
        if len(train_list) == 0 or len(val_list) == 0 :
            print("List empty")
        else:
            max_val_acc = max(val_list)
            epoch_acc = val_list.index(max_val_acc)
            print(f"Optimal point : {epoch_acc} epoch with Val Accuracy {max_val_acc}")
            plt.figure()
            plt.plot(range(len(train_list)), train_list, color='blue', label='Train Accuracy')
            plt.plot(range(len(val_list)), val_list, color='green', label='Valid Accuracy')
            plt.plot(epoch_acc, max_val_acc, marker = 'v', color = 'purple', label='Max Val Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score')
            plt.title('Training Summary : Accuracy Score')
            plt.legend()
            plt.savefig(fig_path)
    else :
        print("Invalid plot type")