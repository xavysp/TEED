import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json

DATASET_NAMES = [
    'BIPED',
    'BIPED-B2',
    'BIPED-B3',
    'BIPED-B5',
    'BIPED-B6',
    'BSDS', # 5
    'BRIND', # 6
    'BSDS300',
    'CID', #8
    'DCD',
    'MDBD', #10
    'PASCAL',
    'NYUD', #12
    'BIPBRI',
    'CLASSIC'
]  # 8

BIPED_mean = [129.510, 129.451,132.230,137.86]

def dataset_info(dataset_name, is_linux=True):
    if is_linux:

        config = {
            'BIPBRI': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair0.lst',
                'train_list2': 'train_pair.lst',
                'test_list': None,
                'data_dir': '/root/workspace/datasets/BIPED',  # mean_rgb
                'data_dir2': '/root/workspace/datasets/BRIND',  # mean_rgb
                'yita': 0.5,
                'mean': [159.510, 159.451,162.230,137.86]
            },
            'BSDS': {
                'img_height': 512, #321
                'img_width': 512, #481
                'train_list': 'train_pair.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/root/workspace/datasets/BSDS',  # mean_rgb
                'yita': 0.5,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },
            'BRIND': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/root/workspace/datasets/BRIND',  # mean_rgb
                'yita': 0.5,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },
            'BSDS300': {
                'img_height': 512, #321
                'img_width': 512, #481
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/root/workspace/datasets/BSDS300',  # NIR
                'yita': 0.5,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },
            'PASCAL': {
                'img_height': 416, # 375
                'img_width': 512, #500
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/root/datasets/PASCAL',  # mean_rgb
                'yita': 0.3,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },
            'CID': {
                'img_height': 512,
                'img_width': 512,
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/root/datasets/CID',  # mean_rgb
                'yita': 0.3,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },
            'NYUD': {
                'img_height': 448,#425
                'img_width': 560,#560
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/root/datasets/NYUD',  # mean_rgb
                'yita': 0.5,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },
            'MDBD': {
                'img_height': 720,
                'img_width': 1280,
                'test_list': 'test_pair.lst',
                'train_list': 'train_pair.lst',
                'data_dir': '/root/workspace/datasets/MDBD',  # mean_rgb
                'yita': 0.3,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },
            'BIPED': {
                'img_height': 720, #720 # 1088
                'img_width': 1280, # 1280 5 1920
                'test_list': 'test_pair.lst',
                'train_list': 'train_pairB5.lst',
                'data_dir': '/root/workspace/datasets/BIPED',  # mean_rgb
                'yita': 0.5,
                'mean':BIPED_mean
                # train_pairB5.lst this for DexiNed and LDC train_pair0.lst
            # ori [159.510, 159.451,162.230,137.86]
            },
            'CLASSIC': {
                'img_height': 512,#
                'img_width': 512,# 512
                'test_list': None,
                'train_list': None,
                'data_dir': 'data',  # mean_rgb
                'yita': 0.5,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },
            'BIPED-B2': {'img_height': 720,  # 720
                         'img_width': 1280,  # 1280
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_rgb.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                         'yita': 0.5,
                         'mean':BIPED_mean},
            'BIPED-B3': {'img_height': 720,  # 720
                         'img_width': 1280,  # 1280
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_rgb.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                         'yita': 0.5,
                         'mean':BIPED_mean},
            'BIPED-B5': {'img_height': 720,  # 720
                         'img_width': 1280,  # 1280
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_rgb.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                         'yita': 0.5,
                        'mean':BIPED_mean},
            'BIPED-B6': {'img_height': 720,  # 720
                         'img_width': 1280,  # 1280
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_rgb.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                         'yita': 0.5,
                         'mean':BIPED_mean},
            'DCD': {
                'img_height': 352, #240
                'img_width': 480,# 360
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/DCD',  # mean_rgb
                'yita': 0.2,
                'mean': [104.007, 116.669, 122.679, 137.86]
            }
        }
    else:
        config = {
            'BIPBRI': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair.lst',
                'train_list2': 'train_pair.lst',
                'test_list': None,
                'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # mean_rgb
                'data_dir2': 'C:/Users/xavysp/dataset/BRIND',  # mean_rgb
                'yita': 0.5,
                'mean': BIPED_mean
            },
            'BSDS': {'img_height': 480,  # 321
                     'img_width': 480,  # 481
                     'test_list': 'test_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/BSDS',  # mean_rgb
                     'yita': 0.5,
                    'mean':[103.939, 116.669, 122.679, 137.86] },
            # [103.939, 116.669, 122.679, 137.86]
            #[159.510, 159.451,162.230,137.86]
            'BRIND': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair2.lst',
                'test_list': 'test_pair.lst',
                'data_dir': 'C:/Users/xavysp/dataset/BRIND',  # mean_rgb
                'yita': 0.5,
                'mean': [104.007, 116.669, 122.679, 137.86]
            },

            'BSDS300': {'img_height': 512,  # 321
                        'img_width': 512,  # 481
                        'test_list': 'test_pair.lst',
                        'data_dir': 'C:/Users/xavysp/dataset/BSDS300',  # NIR
                        'yita': 0.5,
                    'mean': [104.007, 116.669, 122.679, 137.86]},
            'PASCAL': {'img_height': 375,
                       'img_width': 500,
                       'test_list': 'test_pair.lst',
                       'data_dir': 'C:/Users/xavysp/dataset/PASCAL',  # mean_rgb
                       'yita': 0.3,
                    'mean': [104.007, 116.669, 122.679, 137.86]},
            'CID': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_pair.lst',
                    'data_dir': 'C:/Users/xavysp/dataset/CID',  # mean_rgb
                    'yita': 0.3,
                    'mean': [104.007, 116.669, 122.679, 137.86]},
            'NYUD': {'img_height': 425,
                     'img_width': 560,
                     'test_list': 'test_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/NYUD',  # mean_rgb
                     'yita': 0.5,
                    'mean': [104.007, 116.669, 122.679, 137.86]},
            'MDBD': {'img_height': 720,
                         'img_width': 1280,
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_pair.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/MDBD',  # mean_rgb
                         'yita': 0.3,
                         'mean': [104.007, 116.669, 122.679, 137.86]},
            'BIPED': {'img_height': 720,  # 720
                      'img_width': 1280,  # 1280
                      'test_list': 'test_pair.lst',
                      'train_list': 'train_rgb.lst',
                      'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5,
                      'mean':BIPED_mean},
            'BIPED-B2': {'img_height': 720,  # 720
                      'img_width': 1280,  # 1280
                      'test_list': 'test_pair.lst',
                      'train_list': 'train_rgb.lst',
                      'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5,
                      'mean':BIPED_mean},
            'BIPED-B3': {'img_height': 720,  # 720
                      'img_width': 1280,  # 1280
                      'test_list': 'test_pair.lst',
                      'train_list': 'train_rgb.lst',
                      'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5,
                      'mean':BIPED_mean},
            'BIPED-B5': {'img_height': 720,  # 720
                         'img_width': 1280,  # 1280
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_rgb.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                         'yita': 0.5,
                         'mean':BIPED_mean},
            'BIPED-B6': {'img_height': 720,  # 720
                         'img_width': 1280,  # 1280
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_rgb.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                         'yita': 0.5,
                         'mean':BIPED_mean},
            'CLASSIC': {'img_height': 512,
                        'img_width': 512,
                        'test_list': None,
                        'train_list': None,
                        'data_dir': 'data',  # mean_rgb
                        'yita': 0.5,
                        'mean': [104.007, 116.669, 122.679, 137.86]},
            'DCD': {'img_height': 240,
                    'img_width': 360,
                    'test_list': 'test_pair.lst',
                    'data_dir': 'C:/Users/xavysp/dataset/DCD',  # mean_rgb
                    'yita': 0.2,
                    'mean': [104.007, 116.669, 122.679, 137.86]}
        }
    return config[dataset_name]


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None
                 ):
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args = arg
        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = arg.mean_test if len(arg.mean_test) == 3 else arg.mean_test[:3]
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        if self.test_data == "CLASSIC":
            # for single image testing
            images_path = os.listdir(self.data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        else:
            # image and label paths are located in a list file

            if not self.test_list:
                raise ValueError(
                    f"Test list not provided for dataset: {self.test_data}")

            list_name = os.path.join(self.data_root, self.test_list)
            if self.test_data.upper() in ['BIPED', 'BRIND']:

                with open(list_name) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
            else:
                with open(list_name, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]
                pairs = [line.split() for line in files]

                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
        return sample_indices

    def __len__(self):
        return len(self.data_index[0]) if self.test_data.upper() == 'CLASSIC' else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx] if len(self.data_index[0]) > 1 else self.data_index[0][idx - 1]
        else:
            image_path = self.data_index[idx][0]
        label_path = None if self.test_data == "CLASSIC" else self.data_index[idx][1]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        # base dir
        if self.test_data.upper() == 'BIPED':
            img_dir = os.path.join(self.data_root, 'imgs', 'test')
            gt_dir = os.path.join(self.data_root, 'edge_maps', 'test')
        elif self.test_data.upper() == 'CLASSIC':
            img_dir = self.data_root
            gt_dir = None
        else:
            img_dir = self.data_root
            gt_dir = self.data_root

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        if not self.test_data == "CLASSIC":
            label = cv2.imread(os.path.join(
                gt_dir, label_path), cv2.IMREAD_COLOR)
        else:
            label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {(img_height, img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.resize(img, (img_width, img_height))
            gt = None

        # Make images and labels at least 512 by 512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height))  # 512
            gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height))  # 512

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        else:
            img_width = self.args.test_img_width
            img_height = self.args.test_img_height
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        # # For FPS
        # img = cv2.resize(img, (496,320))
        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


class BipedDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 arg=None
                 ):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = arg.mean_train if len(arg.mean_train) == 3 else arg.mean_train[:3]
        self.crop_img = crop_img
        self.arg = arg

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        if self.arg.train_data.lower() == 'biped':

            images_path = os.path.join(data_root,
                                       'edges/imgs',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)
            labels_path = os.path.join(data_root,
                                       'edges/edge_maps',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)

            for directory_name in os.listdir(images_path):
                image_directories = os.path.join(images_path, directory_name)
                for file_name_ext in os.listdir(image_directories):
                    file_name = os.path.splitext(file_name_ext)[0]
                    sample_indices.append(
                        (os.path.join(images_path, directory_name, file_name + '.jpg'),
                         os.path.join(labels_path, directory_name, file_name + '.png'),)
                    )
        else:
            file_path = os.path.join(data_root, self.arg.train_list)
            if self.arg.train_data.lower() == 'bsds':

                with open(file_path, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))
            else:
                with open(file_path) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255.  # for LDC input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w, _ = img.shape
        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else None  # 448# MDBD=480 BIPED=480/400 BSDS=352

        # # for BSDS 352/BRIND
        # if i_w > crop_size and i_h > crop_size:  # later 400, before crop_size
        #     i = random.randint(0, i_h - crop_size)
        #     j = random.randint(0, i_w - crop_size)
        #     img = img[i:i + crop_size, j:j + crop_size]
        #     gt = gt[i:i + crop_size, j:j + crop_size]

        # for BIPED/MDBD
        if i_w> 420 and i_h>420: #before np.random.random() > 0.4
            h,w = gt.shape
            if np.random.random() > 0.4: #before i_w> 500 and i_h>500:

                LR_img_size = crop_size #l BIPED=256, 240 200 # MDBD= 352 BSDS= 176
                i = random.randint(0, h - LR_img_size)
                j = random.randint(0, w - LR_img_size)
                # if img.
                img = img[i:i + LR_img_size , j:j + LR_img_size ]
                gt = gt[i:i + LR_img_size , j:j + LR_img_size ]
            else:
                LR_img_size = 300#208  # l BIPED=208-352, # MDBD= 352-480- BSDS= 176-320
                i = random.randint(0, h - LR_img_size)
                j = random.randint(0, w - LR_img_size)
                # if img.
                img = img[i:i + LR_img_size, j:j + LR_img_size]
                gt = gt[i:i + LR_img_size, j:j + LR_img_size]
                img = cv2.resize(img, dsize=(crop_size, crop_size), )
                gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        else:
            # New addidings
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # # BRIND
        # gt[gt > 0.1] +=0.2#0.4
        # gt = np.clip(gt, 0., 1.)
        # for BIPED
        gt[gt > 0.2] += 0.6  # 0.5 for BIPED
        gt = np.clip(gt, 0., 1.)  # BIPED
        # # for MDBD
        # gt[gt > 0.3] +=0.7#0.4
        # gt = np.clip(gt, 0., 1.)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt

    # two datasets
class bipbriDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 arg=None,
                 extra_inf=None
                 ):
        self.data_root = data_root
        self.data_root2 = extra_inf['data_dir2']
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = arg.mean_train if len(arg.mean_train) == 3 else arg.mean_train[:3]
        self.crop_img = crop_img
        self.arg = arg
        self.extra_info=extra_inf
        self.train_list = arg.train_list
        self.train_list2 = extra_inf['train_list2']

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)

        sample_indices = []
        list_name = os.path.join(self.data_root, self.train_list)
        list_name2 = os.path.join(self.data_root2, self.train_list2)
        with open(list_name) as f:
            files = json.load(f)
        for pair in files:
            tmp_img = pair[0]
            tmp_gt = pair[1]
            sample_indices.append(
                (os.path.join(self.data_root, tmp_img),
                 os.path.join(self.data_root, tmp_gt),))

        with open(list_name2) as f:
            files2 = json.load(f)
        for pair2 in files2:
            tmp_img = pair2[0]
            tmp_gt = pair2[1]
            sample_indices.append(
                (os.path.join(self.data_root2, tmp_img),
                 os.path.join(self.data_root2, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255.  # for LDC input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w, _ = img.shape
        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else None  # 448# MDBD=480 BIPED=480/400 BSDS=352

        # # for BSDS 352/BRIND
        # if i_w > crop_size and i_h > crop_size:  # later 400, before crop_size
        #     i = random.randint(0, i_h - crop_size)
        #     j = random.randint(0, i_w - crop_size)
        #     img = img[i:i + crop_size, j:j + crop_size]
        #     gt = gt[i:i + crop_size, j:j + crop_size]

        # for BIPED/MDBD
        if i_w> 420 and i_h>420: #before np.random.random() > 0.4
            h,w = gt.shape
            if np.random.random() > 0.4: #before i_w> 500 and i_h>500:

                LR_img_size = crop_size #l BIPED=256, 240 200 # MDBD= 352 BSDS= 176
                i = random.randint(0, h - LR_img_size)
                j = random.randint(0, w - LR_img_size)
                # if img.
                img = img[i:i + LR_img_size , j:j + LR_img_size ]
                gt = gt[i:i + LR_img_size , j:j + LR_img_size ]
            else:
                LR_img_size = 256#208  # l BIPED=300(before) # MDBD= 352-480- BSDS= 176-320
                i = random.randint(0, h - LR_img_size)
                j = random.randint(0, w - LR_img_size)
                # if img.
                img = img[i:i + LR_img_size, j:j + LR_img_size]
                gt = gt[i:i + LR_img_size, j:j + LR_img_size]
                img = cv2.resize(img, dsize=(crop_size, crop_size), )
                gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        else:
            # New addidings
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # # BRIND
        # gt[gt > 0.1] +=0.2#0.4
        # gt = np.clip(gt, 0., 1.)
        # for BIPED
        gt[gt > 0.2] += 0.6  # 0.5 for BIPED
        gt = np.clip(gt, 0., 1.)  # BIPED

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt
