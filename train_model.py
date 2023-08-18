import torch
from pathlib import Path
import numpy as np
import os, time, sys, copy, gc
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler

from utils import generatePlots, getModelName, epoch_time
from services import DataGenerator, EarlyStopper

def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()

    total_loss = 0
    correct = 0
    batch = 0
    total = 0
    
    for imgs, labels_y in train_loader:
        imgs = imgs.to(device)
        labels_y = labels_y.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        _, pred = torch.max(output.data, 1)
        loss = criterion(output, labels_y)
        loss.backward()
        total_loss += loss.item() * imgs.size(0)
        correct += torch.sum(pred == labels_y.data)
        total += labels_y.size(0)

        optimizer.step()
        
        batch += 1
        del imgs
        del labels_y
        del output
        gc.collect()
        torch.cuda.empty_cache()
 
    return correct/total, total_loss/len(dataloader)

def evaluate(model, criterion, val_loader, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    batch = 0
    total = 0
    with torch.no_grad():
        for imgs, labels_y in val_loader:
            imgs = imgs.to(device)
            labels_y = labels_y.to(device)

            output = model(imgs)
            _, pred = torch.max(output.data, 1)
            loss = criterion(output, labels_y)
            correct += torch.sum(pred == labels_y.data)
            epoch_loss += loss.item() * imgs.size(0)
            
            total += labels_y.size(0)
            batch += 1
            del imgs
            del labels_y
            del output
            gc.collect()
            torch.cuda.empty_cache()
 
    return correct/total, epoch_loss/len(dataloader)

def train_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]
    datapath = os.path.join(parent, config.datapath)
    if not os.path.exists(os.path.join(parent, 'Checkpoints')):
        os.mkdir(os.path.join(parent, 'Checkpoints'))
    if not os.path.exists(os.path.join(parent, 'Plots & Outputs')):
        os.mkdir(os.path.join(parent, 'Plots & Outputs'))
    
    if not os.path.exists(datapath):
        sys.exit('Data folder not available')
    
    model_name = getModelName(config)
    output = open(os.path.join(parent, 'Plots & Outputs', f'{model_name}.txt'), 'w')

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    print(f"Device Type : {device}")
    output.write(f"Device Type : {device}\n")

    train_folder = ImageFolder(os.path.join(datapath, 'train'), transform = transform)
    val_folder = ImageFolder(os.path.join(datapath, 'test'), transform = transform)

    train_loader = DataLoader(train_folder, batch_size = config.batch_size, shuffle = True)
    val_loader = DataLoader(val_folder, batch_size = config.batch_size)

    print(f"Number of batches in Train Loader : {len(train_loader)}\nNumber of batches in Validation loader : {len(val_loader)}")
    output.write(f"Number of batches in Train Loader : {len(train_loader)}\nNumber of batches in Validation loader : {len(val_loader)}\n")

    model = resnet34(pretrained = False)
    _ = model.to(device)
    best_model = None

    criterion = CrossEntropyLoss()
    criterion.to(device)

    optimizer = Adam(model.parameters(), lr = config.learning_rate, weight_decay = 0.0004)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
    earlystopper = EarlyStopper(patience = config.patience)

    c = 0
    best_valid_loss = np.inf
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    start = time.time()
    for epoch in range(config.epochs):
        print(f"\nEpoch: {epoch+1:02}\tlearning rate : {scheduler.get_last_lr()}\n")
        output.write(f'\nEpoch: {epoch+1:02}\tlearning rate : {scheduler.get_last_lr()}\n\n')

        start_time = time.time()

        train_acc, train_loss = train_epoch(model, criterion, optimizer, train_loader, device)
        val_acc, val_loss = evaluate(model, criterion, val_loader, device)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        epoch_hr, epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        print(f"Elapsed time : {epoch_hr}h {epoch_mins}m {epoch_secs}s")
        print(f"Train F1-score : {train_f1:.4f}\tTrain Loss : {train_loss:.4f}")
        print(f"Validation F1-score : {val_f1:.4f}\tValidation Loss : {val_loss:.4f}")
        output.write(f"Elapsed time : {epoch_hr}h {epoch_mins}m {epoch_secs}s\nTrain F1-score : {train_f1:.4f}\tTrain Loss : {train_loss:.4f}\nValidation F1-score : {val_f1:.4f}\tValidation Loss : {val_loss:.4f}\n")

        c += 1

        if valid_loss < best_valid_loss :
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model.state_dict())
            print(f"Model recorded with Validation loss : {val_loss}\n")
            output.write(f"Model recorded with Validation loss : {val_loss}\n\n")
            c = 0
        
        if c > 3:
            scheduler.step()
            c = 0
        
        if earlystopper.early_stop(val_loss) :
            print(f"Model is not improving. Quitting ...")
            output.write(f"Model is not improving. Quitting ...\n")
        
        torch.cuda.empty_cache()

    end = time.time()
    train_h, train_m, train_s = epoch_time(start, end)
    print(f"Total training time : {train_h}hrs. {train_m}mins. {train_s}s")
    output.write(f"Total training time : {train_h}hrs. {train_m}mins. {train_s}s\n")
    print(f"For inference, put the model name in 'inference_config.yaml' file\n-> model_name : {model_name}\n")

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), os.path.join(parent, 'Checkpoints', f"{model_name}.pth"))

    if device == 'cuda':
        train_acc_list = [i.to('cpu') for i in train_acc_list]
        val_acc_list = [i.to('cpu') for i in val_acc_list]

    plot_path = os.path.join(parent, 'Plots & Outputs', f'accuracy_plot_{model_name}.jpg')
    generatePlots(train_acc_list, val_acc_list, plot_path, plot_type='acc')
    plot_path = os.path.join(parent, 'Plots & Outputs', f'loss_plot_{model_name}.jpg')
    generatePlots(train_loss_list, val_loss_list, plot_path, plot_type = 'loss')