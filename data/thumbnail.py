"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob
import pandas as pd
import sys
import numpy as np

class Thumbnail(datasets.ImageFolder):

    base_folder = 'thumbnail'
    train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    val_list = [0]
    test_list = [0]

    def __init__(self, root=MyPath.db_root_dir('thumbnail'), split='train', transform=None):
        super(Thumbnail, self).__init__(root=os.path.join(root, split), transform=None)
        self.root = os.path.join(root, split)
        self.transform = transform 
        self.split = split
        self.resize = tf.Resize(96)  # original image size: 164 --> 96
        self.classes = {
                            'Action': 0, 'All Ages': 1, 'Animals': 2, 'Comedy': 3, 'Crime/Mystery': 4,
                            'Drama': 5, 'Fantasy': 6, 'Heartwarming': 7, 'Historical': 8, 'Horror': 9,
                            'Informative': 10, 'Inspirational': 11, 'Post-apocalyptic': 12, 'Romance': 13,
                            'School': 14, 'Sci-fi': 15, 'Short story': 16, 'Slice of life': 17, 'Sports': 18,
                            'Superhero': 19, 'Supernatural': 20, 'Thriller': 21, 'Zombies': 22
                        }

        if self.split == 'train':
            current_list = self.train_list
        elif self.split == 'val':
            current_list = self.val_list
        else:
            current_list = self.test_list

        self.data_loc = []
        self.title_no = []
        self.genre = []

        for i in current_list:
            file_path = os.path.join(self.root, f'{i}_after.csv')
            file = pd.read_csv(file_path, index_col=None)

            loc = [os.path.join(self.root, str(i), f'{n}.jpg') for n in file['number']]

            self.data_loc.extend(loc)
            self.genre.extend(file['genre'])
            self.title_no.extend(file['number'])

            assert len(self.data_loc) == len(self.data_loc) == len(self.title_no), "Lengths are not the same!"

    def __len__(self):
        return len(self.data_loc)


    def __getitem__(self, index):
        # path = os.path.join(self.main_dir, self.total_imgs[index])
        path = self.data_loc[index]
        img = Image.open(path).convert("RGB")
        im_size_ori = img.size
        img = self.resize(img)  # whether or not to resize the image
        im_size = img.size

        genre = self.genre[index]
        target = self.classes[genre]
        title_no = self.title_no[index]

        if self.transform is not None:
            img = self.transform(img)

        out = {
            'image': img,
            'target': target,
            'meta': {
                'im_size': im_size,
                'im_size_ori': im_size_ori,
                'index': index,
                'genre': genre,
                'title_no': title_no
                }
            }

        return out

    def get_image(self, index):
        path = self.data_loc[index]
        img = Image.open(path).convert("RGB")
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img