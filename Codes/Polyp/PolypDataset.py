# -*-ing:utf-8-*-
import os
import torch
import random
import numpy as np
import ml_collections
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_rot_flip(image, label, prob):
    p = random.uniform(0, 1)
    if prob > p:
        k = np.random.randint(1, 4)
        image = np.rot90(image, k, axes=(0, 1))
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label, prob):
    p = random.uniform(0, 1)
    if prob > p:
        angle = np.random.randint(-45, 45)
        image = Image.fromarray(np.uint8(image))
        label = Image.fromarray(np.uint8(label))
        image.rotate(angle, resample=Image.BILINEAR)
        label.rotate(angle)
    return np.array(image), np.array(label)

def weak_data_augmentation(image, label, prob):
    image, label = random_rot_flip(image, label, prob)
    image, label = random_rotate(image, label, prob)
    return image, label

# read RGB images or gray images
def binary_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")


def rgb_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class PylopDataset(data.Dataset):
    def __init__(self, config):
        self.mode = config.mode
        self.dir = config.dir
        self.image_shape = config.image_shape
        self.data_augmentation = config.data_augmentation
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.496, 0.311, 0.226],
                                 std=[0.310, 0.237, 0.211])
        ])

        if self.data_augmentation:
            self.prob = config.data_aug_prob

        self.dir_list = os.listdir(self.dir)
        if self.dir_list[0] == "images":
            self.image_dir = os.path.join(self.dir, self.dir_list[0])
            self.gt_dir = os.path.join(self.dir, self.dir_list[1])
        else:
            self.image_dir = os.path.join(self.dir, self.dir_list[1])
            self.gt_dir = os.path.join(self.dir, self.dir_list[0])

        self.image_list = [os.path.join(self.image_dir, image_path) for image_path in os.listdir(self.image_dir)]
        self.gt_list = [os.path.join(self.gt_dir, gt_path) for gt_path in os.listdir(self.gt_dir)]
        self.image_name = os.listdir(self.image_dir)
        self.gt_name = os.listdir(self.gt_dir)

        self.image_list.sort()
        self.gt_list.sort()
        self.image_name.sort()
        self.gt_name.sort()

        assert len(self.image_list) == len(self.gt_list), "images and labels is not equal"
        self.size = len(self.image_list)

    def __getitem__(self, item):
        image = rgb_loader(self.image_list[item])
        gt = binary_loader(self.gt_list[item])

        image_name = self.image_name[item]
        gt_name = self.gt_name[item]

        if np.array(image).shape[-2] != self.image_shape:
            image = image.resize((self.image_shape, self.image_shape), Image.BILINEAR)
            gt = gt.resize((self.image_shape, self.image_shape), Image.NEAREST)

        image = np.array(image)
        gt = np.array(gt)

        if gt.ndim == 3:
            gt = gt[..., 0]

        if self.mode == "train":
            if self.data_augmentation:
                image, gt = weak_data_augmentation(image, gt, prob=self.prob)

        gt[gt != 0] = 1

        image = self.image_transforms(image).float()
        gt = torch.from_numpy(gt).unsqueeze(0).long()
        sample = {"image": image, "label": gt}
        return sample

    def __len__(self):
        return self.size


Polyp_train_config = ml_collections.ConfigDict()
Polyp_train_config.dir = r"F:\2024_1_6_DECTNet_RevisedVersion\Datasets\Polyp\TrainDataset"
Polyp_train_config.mode = "train"
Polyp_train_config.data_augmentation = True
Polyp_train_config.image_shape = 224
Polyp_train_config.data_aug_prob = 0.3

Polyp_CVC_300_config = ml_collections.ConfigDict()
Polyp_CVC_300_config.dir = r"F:\2024_1_6_DECTNet_RevisedVersion\Datasets\Polyp\TestDataset\CVC-300"
Polyp_CVC_300_config.mode = "test"
Polyp_CVC_300_config.data_augmentation = False
Polyp_CVC_300_config.image_shape = 224
Polyp_CVC_300_config.data_aug_prob = None

Polyp_CVC_ColonDB_config = ml_collections.ConfigDict()
Polyp_CVC_ColonDB_config.dir = r"F:\2024_1_6_DECTNet_RevisedVersion\Datasets\Polyp\TestDataset\CVC-ColonDB"
Polyp_CVC_ColonDB_config.mode = "test"
Polyp_CVC_ColonDB_config.data_augmentation = False
Polyp_CVC_ColonDB_config.image_shape = 224
Polyp_CVC_ColonDB_config.data_aug_prob = None

Polyp_CVC_ClinicDB_config = ml_collections.ConfigDict()
Polyp_CVC_ClinicDB_config.dir = r"F:\2024_1_6_DECTNet_RevisedVersion\Datasets\Polyp\TestDataset\CVC-ClinicDB"
Polyp_CVC_ClinicDB_config.mode = "test"
Polyp_CVC_ClinicDB_config.data_augmentation = False
Polyp_CVC_ClinicDB_config.image_shape = 224
Polyp_CVC_ClinicDB_config.data_aug_prob = None

Polyp_Kvasir_config = ml_collections.ConfigDict()
Polyp_Kvasir_config.dir = r"F:\2024_1_6_DECTNet_RevisedVersion\Datasets\Polyp\TestDataset\Kvasir"
Polyp_Kvasir_config.mode = "test"
Polyp_Kvasir_config.data_augmentation = False
Polyp_Kvasir_config.image_shape = 224
Polyp_Kvasir_config.data_aug_prob = None

Polyp_ETIS_LaribPolypDB_config = ml_collections.ConfigDict()
Polyp_ETIS_LaribPolypDB_config.dir = r"F:\2024_1_6_DECTNet_RevisedVersion\Datasets\Polyp\TestDataset\ETIS-LaribPolypDB"
Polyp_ETIS_LaribPolypDB_config.mode = "test"
Polyp_ETIS_LaribPolypDB_config.data_augmentation = False
Polyp_ETIS_LaribPolypDB_config.image_shape = 224
Polyp_ETIS_LaribPolypDB_config.data_aug_prob = None

polyp_train_dataset = PylopDataset(Polyp_train_config)
polyp_cvc_300_dataset = PylopDataset(Polyp_CVC_300_config)
polyp_cvc_colon_dataset = PylopDataset(Polyp_CVC_ColonDB_config)
polyp_cvc_clinic_dataset = PylopDataset(Polyp_CVC_ClinicDB_config)
polyp_kvasir_dataset = PylopDataset(Polyp_Kvasir_config)
polyp_etis_larib_dataset = PylopDataset(Polyp_ETIS_LaribPolypDB_config)

if __name__ == "__main__":
    print(len(polyp_train_dataset))
    print(len(polyp_cvc_300_dataset))
    print(len(polyp_cvc_colon_dataset))
    print(len(polyp_cvc_clinic_dataset))
    print(len(polyp_kvasir_dataset))
    print(len(polyp_etis_larib_dataset))

    for i in polyp_cvc_300_dataset:
        print(i['image'].shape)
    for i in polyp_cvc_colon_dataset:
        print(i['image'].shape)
    for i in polyp_cvc_clinic_dataset:
        print(i['image'].shape)
    for i in polyp_kvasir_dataset:
        print(i['image'].shape)
    for i in polyp_etis_larib_dataset:
        print(i['image'].shape)





