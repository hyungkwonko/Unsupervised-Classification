import os
import torchvision.datasets as datasets
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms
from glob import glob

class Sample(datasets.ImageFolder):

    def __init__(self, root=MyPath.db_root_dir('samples'), transform=None):
        super(Sample, self).__init__(root=root, transform=None)
        self.root = root
        self.transform = transform 
        self.resize = transforms.Compose([
                        transforms.Resize(164),
                        transforms.CenterCrop(96),
                    ])
        self.data_loc = glob(os.path.join(self.root, '0', '*.jpg'))

    def __len__(self):
        return len(self.data_loc)

    def __getitem__(self, index):
        path = self.data_loc[index]
        img = Image.open(path).convert("RGB")
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'path': path}

        return out

    def get_image(self, index):
        path = self.data_loc[index]
        # img = Image.open(path).convert("RGB")
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img