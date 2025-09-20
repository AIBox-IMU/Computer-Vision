import torch
import random
import numpy as np

from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

import copy
import os
import random
import numpy as np
seed = 1024
# random.seed(seed) # python的随机性
# np.random.seed(seed) # np的随机性
# os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，为了禁止hash随机化
# 数据增强
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        # mask = np.array(mask).astype(np.float32)
        mask = np.array(mask).astype(np.float32) / 255.0

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask = torch.from_numpy(mask).float()

        return {"image": (img1, img2), "label": mask}

# 这个类实现了一个随机水平翻转的功能，概率为50%。__call__
# 方法接受一个包含两张图像和一个掩模的样本，如果随机数小于0.5，
# 则对图像和掩模进行水平翻转。处理后的数据以相同的格式返回
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": (img1, img2), "label": mask}

 # 则对图像和掩模进行竖直翻转
class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {"image": (img1, img2), "label": mask}

# 从样本中提取两张图像和一个掩模，接着生成一个随机旋转角度
# （在 -self.degree 到 self.degree 之间），然后对图像进行双线性插值旋转，
# 对掩模进行最近邻插值旋转
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        if random.random() < 0.5:
            rotate_degree = random.uniform(-1 * self.degree, self.degree)
            img1 = img1.rotate(rotate_degree, Image.BILINEAR)
            img2 = img2.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {"image": (img1, img2), "label": mask}


class Shift(object):
    # def __init__(self, size):
    #     self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]

        Rx = random.randint(-32, 32)
        Rx = Rx + 256 if Rx < 0 else Rx
        Ry = random.randint(-32, 32)
        Ry = Ry + 256 if Ry < 0 else Ry

        img1_p = copy.deepcopy(img1)
        img2_p = copy.deepcopy(img2)
        mask_p = copy.deepcopy(mask)

        img1.paste(img1_p, (Rx, Ry))
        img1.paste(img1_p, (Rx - 256, Ry - 256))
        img1.paste(img1_p, (Rx - 256, Ry))
        img1.paste(img1_p, (Rx, Ry - 256))

        img2.paste(img2_p, (Rx, Ry))
        img2.paste(img2_p, (Rx - 256, Ry - 256))
        img2.paste(img2_p, (Rx - 256, Ry))
        img2.paste(img2_p, (Rx, Ry - 256))

        mask.paste(mask_p, (Rx, Ry))
        mask.paste(mask_p, (Rx - 256, Ry - 256))
        mask.paste(mask_p, (Rx - 256, Ry))
        mask.paste(mask_p, (Rx, Ry - 256))

        return {"image": (img1, img2), "label": mask}


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """

    returnImage = image

    random_factor = np.random.rand() * 0.7 + 0.8  # 随机因子 0.8~1.5
    returnImage = ImageEnhance.Color(returnImage).enhance(
        random_factor
    )  # 调整图像的饱和度

    random_factor = np.random.rand() * 0.7 + 0.8  # 随机因子 0.8~1.5
    returnImage = ImageEnhance.Brightness(returnImage).enhance(
        random_factor
    )  # 调整图像的亮度

    random_factor = np.random.rand() * 0.7 + 0.8  # 随机因子 0.8~1.5
    returnImage = ImageEnhance.Contrast(returnImage).enhance(
        random_factor
    )  # 调整图像对比度

    random_factor = np.random.rand() * 2.2 + 0.8  # 随机因子 0.8~3.0
    returnImage = ImageEnhance.Sharpness(returnImage).enhance(
        random_factor
    )  # 调整图像锐度

    return returnImage


class RandomColor(object):
    # def __init__(self, size):
    #     self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]

        img1 = randomColor(img1)
        img2 = randomColor(img2)

        return {"image": (img1, img2), "label": mask}
class ResizeImgeSize(object):
    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        n = 256
        if img1.size != (n, n):
            img1 = img1.resize((n, n), Image.Resampling.LANCZOS)
            img1 = img1.convert('RGB')
            # 检查并调整 img2 的尺寸
        if img2.size != (n, n):
            img2 = img2.resize((n, n), Image.Resampling.LANCZOS)
            img2 = img2.convert('RGB')

            # 检查并调整 mask 的尺寸
        if mask.size != (n, n):
            mask = mask.resize((n, n), Image.NEAREST)
        mask = mask.convert('L')

        return {"image": (img1, img2), "label": mask}

"""
We don't use Normalize here, because it will bring negative effects.
the mask of ground truth is converted to [0,1] in ToTensor() function.
"""
train_transforms = transforms.Compose(
    [
        ResizeImgeSize(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomColor(),
        Shift(),
        RandomRotate(180),
        ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        ResizeImgeSize(),
        ToTensor()
    ]
)
test_transforms = transforms.Compose(
    [
        ResizeImgeSize(),
        ToTensor()
    ]
)
