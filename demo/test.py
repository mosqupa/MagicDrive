import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from torchvision import transforms
from mmdet3d.datasets import build_dataset

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from magicdrive.runner.utils import concat_6_views
from magicdrive.misc.test_utils import (
    prepare_all, run_one_batch
)
from magicdrive.misc.common import load_module
from magicdrive.unet_2d_condition_multiview_new import UNet2DConditionModelMultiview
from magicdrive.networks.resnet import ResNet34Half, HiddenImageConv, HiddenImageConvPost

transparent_bg = True
target_map_size = 400
# target_map_size = 800


def output_func(x): return concat_6_views(x)
# def output_func(x): return concat_6_views(x, oneline=True)
# def output_func(x): return img_concat_h(*x[:3])


@hydra.main(version_base=None, config_path="../configs",
            config_name="test_config")
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    output_dir = to_absolute_path(cfg.resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml"))
    current_overrides = HydraConfig.get().overrides.task

    # getting the config name of this job.
    config_name = HydraConfig.get().job.config_name
    # concatenating the original overrides with the current overrides
    overrides = original_overrides + current_overrides
    # compose a new config from scratch
    cfg = hydra.compose(config_name, overrides=overrides)
    cfg.runner.validation_index = "demo"

    #### setup everything ####
    pipe, val_dataloader, weight_dtype = prepare_all(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))

    #### start ####
    total_num = 0
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    for val_input in val_dataloader:
        return_tuples = run_one_batch(cfg, pipe, val_input, weight_dtype,
                                      transparent_bg=transparent_bg,
                                      map_size=target_map_size)

        for map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list in zip(*return_tuples):
            # save map
            map_img.save(os.path.join(cfg.log_root, f"{total_num}_map.png"))

            # save ori
            if ori_imgs is not None:
                ori_img = output_func(ori_imgs)
                ori_img.save(os.path.join(cfg.log_root, f"{total_num}_ori.png"))
            # save gen
            for ti, gen_imgs in enumerate(gen_imgs_list):
                gen_img = output_func(gen_imgs)
                gen_img.save(os.path.join(
                    cfg.log_root, f"{total_num}_gen{ti}.png"))

            if cfg.show_box:
                # save ori with box
                if ori_imgs_wb is not None:
                    ori_img_with_box = output_func(ori_imgs_wb)
                    ori_img_with_box.save(os.path.join(
                        cfg.log_root, f"{total_num}_ori_box.png"))
                # save gen with box
                for ti, gen_imgs_wb in enumerate(gen_imgs_wb_list):
                    gen_img_with_box = output_func(gen_imgs_wb)
                    gen_img_with_box.save(os.path.join(
                        cfg.log_root, f"{total_num}_gen{ti}_box.png"))

            total_num += 1

        # update bar
        progress_bar.update(cfg.runner.validation_times)

def get_unet():
    unet_path = "magicdrive-log/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/unet"
    unet_config = "magicdrive.networks.unet_2d_condition_multiview.UNet2DConditionModelMultiview"
    unet_cls = load_module(unet_config)
    unet = unet_cls.from_pretrained(
            unet_path, torch_dtype=torch.float16)

def get_unet_addon():
    unet_addon_path = "magicdrive-log/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/unet_addon"
    unet_addon_config = "magicdrive.networks.unet_2d_condition_multiview.UNet2DConditionModelMultiview"
    unet_addon_cls = load_module(unet_addon_config)
    unet_addon = unet_addon_cls.from_pretrained(
            unet_addon_path, torch_dtype=torch.float16)
    with open("unet_addon_structure.txt", "w") as f:
        print(unet_addon, file=f)

def get_controlnet():
    controlnet_path = "magicdrive-log/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/controlnet"
    controlnet_config = "magicdrive.networks.unet_addon_rawbox.BEVControlNetModel"
    model_cls = load_module(controlnet_config)
    controlnet = model_cls.from_pretrained(
        controlnet_path, torch_dtype=torch.float16)
    with open("controlnet_structure.txt", "w") as f:
        print(controlnet, file=f)

@hydra.main(version_base=None, config_path="../configs",
            config_name="test_config_new")
def get_train_config(cfg: DictConfig):
    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
    OmegaConf.save(config=cfg, f="check.yaml", resolve=True)

def add_conv_to_pretrained_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    unet = UNet2DConditionModelMultiview.from_pretrained("magicdrive-log/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/unet")
    unet.use_virtual_image_encoder = True  # Enable virtual image encoder
    unet.resnet = ResNet34Half().to(device, dtype=dtype)
    unet.hidden_image_conv = HiddenImageConv().to(device, dtype=dtype)
    unet.hidden_image_conv_post = HiddenImageConvPost().to(device, dtype=dtype)
    unet.virtual_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    save_dir = "magicdrive-log/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/unet_addon"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(save_dir, "unet_addon_full.pth"))
    print(f"✅ 成功保存至 {save_dir}/unet_addon_full.pth")

def get_train_config1():
    base_config = OmegaConf.load("configs/tools_train_config.yaml")
    exp_config = OmegaConf.load("configs/exp/224x400.yaml")
    runner_config = OmegaConf.load("configs/runner/debug.yaml")

    # 合并配置，后者覆盖前者
    merged = OmegaConf.merge(base_config, exp_config, runner_config)

    # 保存为完整配置文件
    OmegaConf.save(config=merged, f="configs/tools_train_config_full.yaml")

def check_train_dataset():
    cfg = OmegaConf.load("configs/tools_train_config_full.yaml")
    train_dataset = build_dataset(OmegaConf.to_container(cfg.dataset.data.train, resolve=True))
    print(len(train_dataset.data_infos))  # 或 print(self.val_dataset)
    print(type(train_dataset)) 
    print(len(train_dataset)) # 323

def check_val_dataset():
    cfg = OmegaConf.load("configs/tools_train_config_full.yaml")
    val_dataset = build_dataset(OmegaConf.to_container(cfg.dataset.data.val, resolve=True))
    print(len(val_dataset.data_infos))  # 或 print(self.val_dataset)
    print(type(val_dataset))
    print(len(val_dataset)) # 81
    print(val_dataset[0].keys())
    # dict_keys(['img', 
    # 'gt_bboxes_3d', 
    # 'gt_labels_3d', 
    # 'gt_masks_bev', 
    # 'gt_aux_bev', 
    # 'camera_intrinsics', 
    # 'lidar2ego', 
    # 'lidar2camera', 
    # 'camera2lidar', 
    # 'lidar2image', 
    # 'img_aug_matrix', 
    # 'metas'])
    print(val_dataset[0]['img'].data.shape)  # torch.Size([6, 3, 224, 400])
    print(val_dataset[0]['gt_bboxes_3d'])

if __name__ == "__main__":
    main()