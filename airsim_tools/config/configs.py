from pathlib import Path
import numpy as np

from phd_utils.airsim_tools.save_dataset import AirsimSaverConfig


# Coordinates in Unreal with -z and m instead of cm. Then, transformed to standard coordinates.
player_start = [0.286, -1.332, -1.5677]
map_center = [-0.146, 0.073, -1.4]
base_pose = [map_center[0]-player_start[0], map_center[1]-player_start[1], map_center[2]-player_start[2]]

config = AirsimSaverConfig(
    save_dir=Path("./nerfstudio/"),
    name="airsim",
    mode="from_function",
    function="spherical",
    function_parameters={"radius": np.arange(1, 2, 1), "theta": np.arange(10,90,20), "phi": np.arange(0, 360, 30)},
    cameras=["0"],
    sensors=["poses", "rgb", "depth", "semantic"],
    save_format="colmap",
    origin_transform=np.array(base_pose),
    orientation_transform=180,
    semantic_map=None,
)
