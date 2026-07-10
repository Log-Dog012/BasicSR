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
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        lq = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE)

        if gt is None or lq is None or gt.size == 0 or lq.size == 0:
            return self.__getitem__(np.random.randint(0, len(self)))

        gt = gt.astype(np.float32) / 255.0
        lq = lq.astype(np.float32) / 255.0

        # 防御：图片太小则跳过
        gt_size = self.opt.get('gt_size', 192)
        scale = self.opt.get('scale', 4)
        lr_size = gt_size // scale
        if gt.ndim != 2 or lq.ndim != 2:
            return self.__getitem__(np.random.randint(0, len(self)))
        if gt.shape[0] < gt_size or gt.shape[1] < gt_size:
            return self.__getitem__(np.random.randint(0, len(self)))
        if lq.shape[0] < lr_size or lq.shape[1] < lr_size:
            return self.__getitem__(np.random.randint(0, len(self)))

        # 加通道维度 (H,W) → (H,W,1)
        gt = gt[:, :, np.newaxis]
        lq = lq[:, :, np.newaxis]

        # 随机裁剪（仅训练阶段）
        if self.opt.get('phase', 'train') == 'train':
            gt, lq = self._paired_random_crop(gt, lq, gt_size, scale=scale)

            # 验证裁剪结果
            if gt.shape[0] == 0 or gt.shape[1] == 0 or lq.shape[0] == 0 or lq.shape[1] == 0:
                return self.__getitem__(np.random.randint(0, len(self)))

            # 翻转 & 旋转
            gt, lq = augment([gt, lq],
                             self.opt.get('use_hflip', True),
                             self.opt.get('use_rot', True))

        # numpy → tensor（img2tensor 内部会做 transpose(2,0,1)，这里传入 (H,W,1) 即可）
        from basicsr.utils.img_util import img2tensor
        gt_tensor = img2tensor(gt, bgr2rgb=False, float32=True)
        lq_tensor = img2tensor(lq, bgr2rgb=False, float32=True)

        return {'lq': lq_tensor, 'gt': gt_tensor, 'lq_path': lq_path, 'gt_path': gt_path}

    @staticmethod
    def _paired_random_crop(gt, lq, gt_size, scale):
        """配对随机裁剪（与原版 PairedImageDataset 一致）
        先在 LQ 上随机起点，再映射到 HR
        """
        h_lq, w_lq = lq.shape[0], lq.shape[1]
        lr_size = gt_size // scale

        # 在 LQ 上随机选起点
        top = np.random.randint(0, max(1, h_lq - lr_size + 1))
        left = np.random.randint(0, max(1, w_lq - lr_size + 1))

        # 裁 LQ
        lq = lq[top:top + lr_size, left:left + lr_size, ...]

        # 映射到 HR 起点并裁 HR
        top_gt = int(top * scale)
        left_gt = int(left * scale)
        gt = gt[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]

        return gt, lq


def augment(imgs, hflip=True, rot=True):
    """数据增强：水平翻转、垂直翻转、90度旋转"""
    do_hflip = hflip and np.random.random() < 0.5
    do_vflip = hflip and np.random.random() < 0.5
    do_rot = rot and np.random.random() < 0.5

    if do_hflip:
        imgs = [v[:, ::-1].copy() for v in imgs]
    if do_vflip:
        imgs = [v[::-1, :].copy() for v in imgs]
    if do_rot:
        imgs = [v.transpose(1, 0, 2).copy() for v in imgs]

    return imgs
