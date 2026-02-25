import numpy as np
from PIL import Image
# import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random


class resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
        if 'gt' in sample.keys():
            sample['gt'] = sample['gt'].resize(self.size, Image.BILINEAR)
        if 'mask' in sample.keys():
            sample['mask'] = sample['mask'].resize(self.size, Image.BILINEAR)

        return sample

class random_scale_crop:
    def __init__(self, range=[0.75, 1.25]):
        self.range = range

    def __call__(self, sample):
        scale = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]
        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'gt', 'contour']:
                    base_size = sample[key].size

                    scale_size = tuple((np.array(base_size) * scale).round().astype(int))
                    sample[key] = sample[key].resize(scale_size)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

        return sample

class random_flip:
    def __init__(self, lr=True, ud=True):
        self.lr = lr
        self.ud = ud

    def __call__(self, sample):
        lr = np.random.random() < 0.5 and self.lr is True
        ud = np.random.random() < 0.5 and self.ud is True

        for key in sample.keys():
            if key in ['image', 'gt', 'contour']:
                sample[key] = np.array(sample[key])
                if lr:
                    sample[key] = np.fliplr(sample[key])
                if ud:
                    sample[key] = np.flipud(sample[key])
                sample[key] = Image.fromarray(sample[key])

        return sample

class random_rotate:
    def __init__(self, range=[0, 360], interval=1):
        self.range = range
        self.interval = interval

    def __call__(self, sample):
        rot = (np.random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'gt', 'contour']:
                    base_size = sample[key].size

                    sample[key] = sample[key].rotate(rot, expand=True)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

        return sample

class random_image_enhance:
    def __init__(self, methods=['contrast', 'brightness', 'sharpness', 'color']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)
        if 'color' in methods:
            self.enhance_method.append(ImageEnhance.Color)
    
    def __call__(self, sample):
        image = sample['image']
        np.random.shuffle(self.enhance_method)

        for method in self.enhance_method:
            enhancer = method(image)
            factor = float(1 + np.random.random() / 10)
            if method == ImageEnhance.Contrast:
                factor = random.randint(8, 12) / 10.0
            if method == ImageEnhance.Color:
                factor = random.randint(8, 12) / 10.0
            if method == ImageEnhance.Sharpness:
                factor = random.randint(0, 30) / 10.0
            if method == ImageEnhance.Brightness:
                factor = random.randint(8, 12) / 10.0
            image = enhancer.enhance(factor)
        sample['image'] = image

        return sample

    # def __call__(self, sample):
    #     image = sample['image']
    #     np.random.shuffle(self.enhance_method)

    #     for method in self.enhance_method:
    #         if np.random.random() > 0.5:
    #             enhancer = method(image)
    #             factor = float(1 + np.random.random() / 10)
    #             image = enhancer.enhance(factor)
    #     sample['image'] = image

    #     return sample


class random_gaussian_blur:
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        if np.random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        sample['image'] = image

        return sample

class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        image= sample['image']

        sample['image'] = np.array(image, dtype=np.float32)
        
        return sample

class normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image /= 255
        image -= self.mean
        image /= self.std

        sample['image'] = image

        return sample

class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        
        sample['image'] = image

        return sample
