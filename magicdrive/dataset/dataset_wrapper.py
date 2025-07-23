import os
from glob import glob

import torch
from mmcv.parallel.data_container import DataContainer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes, Box3DMode


class ListSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, dataset, list) -> None:
        self.dataset = dataset
        self.list = list

    def __getitem__(self, idx):
        return self.dataset[self.list[idx]]

    def __len__(self):
        return len(self.list)


class FolderSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, folder) -> None:
        self.dataset = glob(os.path.join(folder, "*.pth"))

    def __getitem__(self, idx):
        data = torch.load(self.dataset[idx])
        mmdet3d_format = {}
        mmdet3d_format['gt_masks_bev'] = data['gt_masks_bev']
        # fmt: off
        # in DataContainer
        mmdet3d_format['img'] = DataContainer(data['img'])
        mmdet3d_format['gt_labels_3d'] = DataContainer(data['gt_labels_3d'])
        mmdet3d_format['camera_intrinsics'] = DataContainer(data['camera_intrinsics'])
        mmdet3d_format['lidar2camera'] = DataContainer(data['lidar2camera'])
        mmdet3d_format['img_aug_matrix'] = DataContainer(data['img_aug_matrix'])
        mmdet3d_format['metas'] = DataContainer(data['metas'])
        # special class
        gt_bboxes_3d = data['gt_bboxes_3d'][:, :7]  # or all, either can work
        mmdet3d_format['gt_bboxes_3d'] = DataContainer(LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR))

        # recompute
        camera2lidar = torch.eye(4, dtype=data['lidar2camera'].dtype)
        camera2lidar = torch.stack([camera2lidar] * len(data['lidar2camera']))
        camera2lidar[:, :3, :3] = data['lidar2camera'][:, :3, :3].transpose(1, 2)
        camera2lidar[:, :3, 3:] = torch.bmm(-camera2lidar[:, :3, :3], data['lidar2camera'][:, :3, 3:])
        mmdet3d_format['camera2lidar'] = DataContainer(camera2lidar)
        mmdet3d_format['lidar2image'] = DataContainer(
            torch.bmm(data['camera_intrinsics'], data['lidar2camera'])
        )
        # fmt: on
        return mmdet3d_format

    def __len__(self):
        return len(self.dataset)

"""
{
    # 1. 多视角图像
    'img': DataContainer(torch.FloatTensor, shape=(6, 3, 224, 400)),
    
    # 2. 3D边界框（已封装为 LiDAR 坐标系下的结构化类）
    'gt_bboxes_3d': DataContainer(
        LiDARInstance3DBoxes,  # box_dim=7, origin=(0.5, 0.5, 0)
        tensor shape=(N, 7)
    ),
    
    # 3. 3D边界框类别
    'gt_labels_3d': DataContainer(torch.LongTensor, shape=(N,)),

    # 4. BEV supervision 目标图
    'gt_masks_bev': torch.FloatTensor, shape=(8, 200, 200),

    # 5. 相机内参
    'camera_intrinsics': DataContainer(torch.FloatTensor, shape=(6, 4, 4)),

    # 6. LiDAR → Camera 的外参矩阵
    'lidar2camera': DataContainer(torch.FloatTensor, shape=(6, 4, 4)),

    # 7. Camera → LiDAR 的外参（反推获得）
    'camera2lidar': DataContainer(torch.FloatTensor, shape=(6, 4, 4)),

    # 8. LiDAR → image（K @ T）
    'lidar2image': DataContainer(torch.FloatTensor, shape=(6, 4, 4)),

    # 9. 图像数据增强矩阵（如 resize、crop）
    'img_aug_matrix': DataContainer(torch.FloatTensor, shape=(6, 4, 4)),

    # 10. 元信息（主要用于生成 prompt）
    'metas': DataContainer(dict)，包含：
        {
            'location': str (如 "boston-seaport"),
            'description': str (如 "dense traffic"),
            'timeofday': str,
            'token': str (nuscenes样本token)
        }
}
"""