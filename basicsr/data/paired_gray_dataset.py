"""
单通道灰度配对图像数据集
继承 PairedImageDataset，在加载时自动转为灰度单通道
"""
import cv2
import numpy as np
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedGrayDataset(PairedImageDataset):
    """灰度配对数据集：读入后转为单通道"""

    def __getitem__(self, index):
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        # 读取灰度图
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        lq = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # 随机裁剪
        gt_size = self.opt.get('gt_size', None)
        if gt_size is not None:
            scale = self.opt.get('scale', 4)
            gt, lq = self._paired_random_crop(gt, lq, gt_size, scale=scale)

        # 翻转 & 旋转
        gt, lq = augment([gt, lq], self.opt['use_hflip'], self.opt['use_rot'])

        # HWC → CHW (单通道)
        gt = gt[np.newaxis, :, :]
        lq = lq[np.newaxis, :, :]

        # numpy → tensor
        from basicsr.utils.img_util import img2tensor
        gt_tensor = img2tensor(gt, bgr2rgb=False, float32=True)
        lq_tensor = img2tensor(lq, bgr2rgb=False, float32=True)

        return {'lq': lq_tensor, 'gt': gt_tensor, 'lq_path': lq_path, 'gt_path': gt_path}

    @staticmethod
    def _paired_random_crop(gt, lq, gt_size, scale):
        """配对随机裁剪：LQ 裁 gt_size//scale，HR 裁 gt_size"""
        h, w = gt.shape[:2]
        lr_size = gt_size // scale
        rnd_h = np.random.randint(0, max(0, h - gt_size))
        rnd_w = np.random.randint(0, max(0, w - gt_size))
        gt = gt[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size]
        lq = lq[rnd_h:rnd_h + lr_size, rnd_w:rnd_w + lr_size]
        return gt, lq


def augment(imgs, hflip=True, rot=True):
    """数据增强"""
    hswap = np.random.uniform(0, 1, 1) < 0.5 if hflip else False
    vswap = np.random.uniform(0, 1, 1) < 0.5 if hflip else False
    rot90 = np.random.uniform(0, 1, 1) < 0.5 if rot else False

    if hswap:
        imgs = [v[:, ::-1].copy() for v in imgs]
    if vswap:
        imgs = [v[::-1, :].copy() for v in imgs]
    if rot90:
        imgs = [v.transpose(1, 0, 2).copy() for v in imgs]

    return imgs
