import torchvision
from PIL import Image
from torch.utils.data import DataLoader

test_dataset = torchvision.datasets.CIFAR10("./CIFAR10dataset", train=True, transform=torchvision.transforms.ToTensor())

test_dataloader = DataLoader(test_dataset)

img = Image.open("D:\\1Pytorch_learn\\VGG16_model\\predict_image\\bird.png")
to_tensor = torchvision.transforms.ToTensor()
img = to_tensor(img)
print(img)

# for data in test_dataloader:
#     img, target = data
#     print(img)