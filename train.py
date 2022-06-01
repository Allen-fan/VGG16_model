import time
import torchvision.datasets
from torch.utils.data import DataLoader
from Model import *
from tqdm import tqdm

from torch.optim.lr_scheduler import ExponentialLR

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10("./CIFAR10dataset", train=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10("./CIFAR10dataset", train=False, transform=transform)

train_dataloader = DataLoader(train_dataset, 128)
val_dataloader = DataLoader(val_dataset, 128)

train_data_size = len(train_dataset)
val_data_size = len(val_dataset)

VGGModel = VGG16()
VGGModel = VGGModel.cuda()

loss_fn =nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

learning_rate = 0.03
optimizer = torch.optim.SGD(VGGModel.parameters(), lr=learning_rate)
ExpLR = ExponentialLR(optimizer, gamma=0.98) #指数衰减

total_train_step = 0
total_val_step = 0
epoch = 20


for i in range(epoch):
    print("----------第{}轮训练----------".format(i + 1))
    time.sleep(0.01)
    #训练开始
    VGGModel.train()
    for data in tqdm(train_dataloader):
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = VGGModel(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            print("训练次数{}，Loss：{}".format(total_train_step, loss.item()))

    ExpLR.step()

    VGGModel.eval()
    total_val_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in val_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = VGGModel(imgs)
            loss = loss_fn(outputs, targets)
            total_val_loss = total_val_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集Loss：{}".format(total_val_loss))
    print("整体测试集accuracy：{}".format(total_accuracy / val_data_size))
    total_val_step = total_val_step + 1

torch.save(VGGModel.state_dict(), "CIFAR10_VGG16.pth")