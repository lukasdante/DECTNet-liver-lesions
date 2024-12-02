# -*-ing:utf-8-*-
import os
import torch
import random
import numpy as np
import ml_collections
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

# data_augmentation
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

class BaseDataset(data.Dataset):
    def __init__(self, config):
        self.mode = config.mode
        self.root_dir = config.root_dir
        self.image_shape = config.image_shape
        self.data_augmentation = config.data_augmentation
        if self.data_augmentation:
            self.prob = config.data_aug_prob

        self.dir_list = os.listdir(self.root_dir)
        print(self.dir_list)
        if self.dir_list[0] == "TrainDataset":
            self.train_dir = os.path.join(self.root_dir, self.dir_list[0])
            self.test_dir = os.path.join(self.root_dir, self.dir_list[1])
        else:
            self.train_dir = os.path.join(self.root_dir, self.dir_list[1])
            self.test_dir = os.path.join(self.root_dir, self.dir_list[0])

        if self.mode == "train":
            self.dir = self.train_dir
            self.dir_list = os.listdir(self.dir)
            if self.dir_list[0] == "images":
                self.image_dir = os.path.join(self.dir, self.dir_list[0])
                print(self.image_dir)
                self.gt_dir = os.path.join(self.dir, self.dir_list[1])
            else:
                self.image_dir = os.path.join(self.dir, self.dir_list[1])
                print(self.image_dir)
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

        else:
            self.dir = self.test_dir
            self.dir_list = os.listdir(self.dir)
            if self.dir_list[0] == "images":
                self.image_dir = os.path.join(self.dir, self.dir_list[0])
                self.gt_dir = os.path.join(self.dir, self.dir_list[1])
            else:
                self.image_dir = os.path.join(self.dir, self.dir_list[1])
                self.gt_dir = os.path.join(self.dir, self.dir_list[0])

            self.image_list = [os.path.join(self.image_dir, image_path) for image_path in os.listdir(self.image_dir)]
            self.gt_list = [os.path.join(self.gt_dir, gt_path) for gt_path in os.listdir(self.gt_dir)]
            self.gt_name = os.listdir(self.gt_dir)
            self.image_name = os.listdir(self.image_dir)

            self.image_list.sort()
            self.gt_list.sort()
            self.image_name.sort()
            self.gt_name.sort()

            assert len(self.image_list) == len(self.gt_list), "images and labels is not equal"
            self.size = len(self.image_list)

    def __getitem__(self, item):
        pass

    def __len__(self):
        return self.size


class MMsDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.808, 0.666, 0.615],
                                 std=[0.150, 0.176, 0.192])
        ])

    def __getitem__(self, item):
        image = rgb_loader(self.image_list[item])
        gt = binary_loader(self.gt_list[item])

        if np.array(image).shape[-2] != self.image_shape:
            image = image.resize((self.image_shape, self.image_shape), Image.BILINEAR)
            gt = gt.resize((self.image_shape, self.image_shape), Image.NEAREST)

        image = np.array(image)
        gt = np.array(gt)

        if self.mode == "train":
            if self.data_augmentation:
                image, gt = weak_data_augmentation(image, gt, prob=self.prob)

        image = self.image_transforms(image).float()
        gt = torch.from_numpy(gt).unsqueeze(0).long()
        sample = {"image": image, "label": gt}
        return sample



MMs_train_config = ml_collections.ConfigDict()
MMs_train_config.root_dir = r"/home/louis/Desktop/dectnet/Codes/FLLs/dataset/root"
MMs_train_config.mode = "train"
MMs_train_config.data_augmentation= False
MMs_train_config.image_shape = 224
MMs_train_config.data_aug_prob = None

MMs_test_config = ml_collections.ConfigDict()
MMs_test_config.root_dir = r"/home/louis/Desktop/dectnet/Codes/FLLs/dataset/root"
MMs_test_config.mode = "test"
MMs_test_config.data_augmentation = False
MMs_test_config.image_shape = 224
MMs_test_config.data_aug_prob = None

mms_train_dataset = MMsDataset(MMs_train_config)
mms_test_dataset = MMsDataset(MMs_test_config)


if __name__ == "__main__":
    print("mms dataset")
    print(len(mms_train_dataset))
    print(len(mms_test_dataset))
    for i in mms_train_dataset:
        print("image:", i['image'].shape)
        print("label:", i['label'].shape)