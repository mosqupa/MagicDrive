import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm

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
    print(output_dir)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml"))
    current_overrides = HydraConfig.get().overrides.task

if __name__ == "__main__":
    main()