import os

from PIL import Image
from torch.utils.data.dataset import Dataset


class Get_imageInfo(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)

    def get_Imagename(self, img_name):
        img_name = img_name.split('.')[0]
        return img_name

    def __getitem__(self, idx):
        image_name = self.img_list[idx]
        image_item_path = os.path.join(self.img_dir, image_name)
        img = Image.open(image_item_path)
        name = self.get_Imagename(image_name)
        return img, name

    def __len__(self):
        return len(self.img_list)