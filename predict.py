import torchvision
from Model import *
from Imageinfo import *

img_dir ="./predict_image"
img_info = Get_imageInfo(img_dir)

pre_class = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat",
             4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship",
             9:"truck"}

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

pre_model = VGG16()
pre_model.load_state_dict(torch.load("./CIFAR10_VGG16.pth"), False)

for i in range(img_info.__len__()):
    img, imgname = img_info.__getitem__(i)
    img = transform(img)
    img = torch.reshape(img, [1, 3, 32, 32])
    with torch.no_grad():
        output = pre_model(img)
    print("target:" + img_info.get_Imagename(imgname) + "," + "predict:" + pre_class.get(output.argmax(1).item()))
