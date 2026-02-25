import os

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import csv
from workspace.VLP.EndoVLP.dataset.custom_transforms import *
import random
from collections import defaultdict
import json

def read_path_text_from_json(file_path):
    """
    从文件读取列表，每行去除换行符。
    参数:
        file_path (str): 文件路径
    返回:
        index_list (list): 文件夹名列表
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f, start=1):
            line = line.strip()
            data = json.loads(line)
            data_list.append(data)
    return data_list


class PolypDataset(data.Dataset):
    def __init__(self, data_file, transform_list, is_transform, is_train=True, testsize=299, debug=False):
        data_list = read_path_text_from_json(data_file)
        self.is_train = is_train

        if debug:
            data_list = data_list[:20] + data_list[-20:]

        self.bags = []
        self.texts = []
        self.folder_names = []
        self.image_nums = 0
        for i, content in enumerate(data_list):
            folder_path = content['image_path']
            folder = folder_path.split('/')[-2] + '/' + folder_path.split('/')[-1]

            images = []
            for f in os.listdir(folder_path):
                if '报告' in f:
                    continue
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG'):
                if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG'):
                    images.append(os.path.join(folder_path, f))

            self.bags.append(images)
            self.texts.append(content['report_text'])
            self.folder_names.append(folder)
            self.image_nums += len(images)

        if is_transform:
            self.transform = self.get_transform(transform_list)
        else:
            self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        print('{} data: cases@ [{}], images@ [{}]'.format('train' if is_train else 'test'), len(self.bags), self.image_nums)
        
    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        images = self.bags[index]
        text = self.texts[index]
        folder_name = self.folder_names[index]

        images_tensor = []
        images_name_tensor = []
        for img_path in images:
            image = Image.open(img_path).convert('RGB')
            if self.is_train:
                sample = {"image": image}
                sample = self.transform(sample)
                images_tensor.append(sample["image"])
            else:
                sample = self.transform(image)
                images_tensor.append(sample)
            images_name_tensor.append(os.path.split(img_path)[-1])

        return torch.stack(images_tensor), torch.tensor(text), torch.stack(images_name_tensor), torch.tensor(folder_name)
    
    def __len__(self):
        return len(self.bags)
