import torch
import torchvision as tv
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os
import torch.nn.functional as F
from tqdm.autonotebook import tqdm


transforms = tv.transforms.Compose([
    tv.transforms.Resize((224,224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = 'PATH'

dataset_train = tv.datasets.ImageFolder(
    root=dataset_path,
    transform=transforms,
)

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    dataset_train, shuffle=True,
    batch_size=batch_size, num_workers=1, drop_last=True
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = tv.models.resnet18()

model.fc = nn.Linear(512, 10)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.6
)


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = model.to(device)
loss_fn = loss_fn.to(device)

if __name__ == "__main__":
    epochs = 10
    loss_epochs_list = []
    acc_epochs_list = []
    for epoch in range(epochs):
        loss_val = 0
        acc_val = 0
        for sample in (pbar := tqdm(train_loader)):
            img, label = sample
            label = F.one_hot(label, 10).float()
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred=model(img)
            
            label = torch.flatten(label)

            print(pred.shape, label.shape)
            pred = F.softmax(pred, dim=1)
            loss = loss_fn(pred, label)

            loss.backward()
            loss_item = loss.item()
            loss_val += loss_item

            optimizer.step()

            acc_current = accuracy(pred.cpu().float(), label.cpu().float())
            acc_val += acc_current

            pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        scheduler.step()
        loss_epochs_list += [loss_val / len(train_loader)]
        acc_epochs_list += [acc_val / len(train_loader)]
        print(loss_epochs_list[-1])
        print(acc_epochs_list[-1])
