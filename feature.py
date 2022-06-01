import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import math

from Model import *


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return image


net = VGG16()
net.load_state_dict(torch.load("./CIFAR10_VGG16.pth"))

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

img = Image.open("./predict_image/dog.png")
img = transform(img)
img = torch.reshape(img, [1, 3, 32, 32])

conv2d_step = 0
maxpool2d_step = 0
relu_step = 0

for i, m in enumerate(net.modules()):
    if i > 1:
        img = m(img)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.ReLU):
            out_img = tensor_to_PIL(img)
            k = math.ceil(math.sqrt(out_img.shape[0]))
            fig, ax = plt.subplots(k, k, figsize=(64, 64))
            # if isinstance(m, nn.Conv2d):conv2d_step += 1
            # else: maxpool2d_step += 1
            # fig.suptitle(str(m) + "   " + str(conv2d_step if isinstance(m, nn.Conv2d) else maxpool2d_step))
            if isinstance(m, nn.Conv2d):
                conv2d_step += 1
                fig.suptitle(str(m) + "   " + str(conv2d_step))
            elif isinstance(m, nn.MaxPool2d):
                maxpool2d_step += 1
                fig.suptitle(str(m) + "   " + str(maxpool2d_step))
            elif isinstance(m, nn.ReLU):
                relu_step += 1
                fig.suptitle(str(m) + "   " + str(relu_step))
            axes = ax.flatten()
            for j in range(len(axes)):
                if j in range(out_img.shape[0]):
                    axes[j].imshow(out_img[j].detach().numpy(), cmap='coolwarm', origin='upper')
                axes[j].axis('off')
            plt.show()

