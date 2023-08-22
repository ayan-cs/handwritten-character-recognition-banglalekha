import torch, os
from torchvision.models import resnet34
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path

def run_inference(config, output):
    parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]

    datapath = os.path.join(parent, config.datapath)
    num_classes = len(os.listdir(os.path.join(parent, datapath, 'test')))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet34(pretrained = False)
    model.fc = nn.Linear(512, num_classes, bias=True)
    model.load_state_dict(torch.load(os.path.join(parent, 'Checkpoints', f"{config.model_name}.pth")))
    _ = model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_folder = ImageFolder(os.path.join(datapath, 'test'), transform = transform)
    test_dl = DataLoader(test_folder, batch_size = config.batch_size)

    total = 0
    correct1 = 0
    with torch.no_grad():
        for (img, label) in test_dl:
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct1 += (predicted == label).sum().item()
            #correct2 += torch.sum(predicted == label.data)
    print(f"Accuracy on Validation set : {correct1/total}")