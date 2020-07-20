import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg


class LabelProcessor:
    """对标签图像的编码"""

    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):     # data process and load.ipynb: 标签编码，返回哈希表
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):

        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class CamvidDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.center_crop(img, label, self.crop_size)

        img, label = self.img_transform(img, label)
        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = sorted([os.path.join(path, img)
                                 for img in files_list])
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        label = np.array(label)  # 以免不是np格式的数据
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose([transforms.ToTensor(
        ), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)

        return img, label


label_processor = LabelProcessor(cfg.class_dict_path)

if __name__ == "__main__":

    TRAIN_ROOT = './CamVid/train'
    TRAIN_LABEL = './CamVid/train_labels'
    VAL_ROOT = './CamVid/val'
    VAL_LABEL = './CamVid/val_labels'
    TEST_ROOT = './CamVid/test'
    TEST_LABEL = './CamVid/test_labels'
    crop_size = (352, 480)
    Cam_train = CamvidDataset([TRAIN_ROOT, TRAIN_LABEL], crop_size)
    Cam_val = CamvidDataset([VAL_ROOT, VAL_LABEL], crop_size)
    Cam_test = CamvidDataset([TEST_ROOT, TEST_LABEL], crop_size)
    for sample in Cam_val:
        img = sample['img']
        label = sample['label']
        print(img.shape)
        print(label[0])
        break
