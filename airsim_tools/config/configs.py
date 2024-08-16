from pathlib import Path
import numpy as np

from airsim_tools.save_dataset import AirsimSaverConfig


# Coordinates in Unreal with -z and m instead of cm. Then, transformed to standard coordinates.
player_start = [0.286, -1.332, -1.5677]
map_center = [-0.146, 0.073, -1.4]
base_pose = [map_center[0]-player_start[0], map_center[1]-player_start[1], map_center[2]-player_start[2]]

config = AirsimSaverConfig(
    save_dir=Path("./nerfstudio/"),
    name=Path("airsim"),
    mode="from_function",
    function="spherical",
    function_parameters={"radius": np.arange(1, 2, 1), "theta": np.arange(5,90,5), "phi": np.arange(0, 360, 10)},
    cameras=["0"],
    data_types=["pose", "rgb", "depth", "semantic"],
    save_format="nerfstudio",
    origin_transform=np.array(base_pose),
    orientation_transform=180,
    semantic_map=None,
)

config = AirsimSaverConfig(
    save_dir=Path("/media/david/datasets/robust_bayesian_semantic"),
    name=Path("house6"),
    mode="from_poses",
    cameras=["0"],
    data_types=["pose", "rgb", "depth", "semantic"],
    save_format="scannet",
    origin_transform=np.array(base_pose),
    orientation_transform=0.0,
    semantic_map=None,
    semantic_config=[["Bottle", 5], ["bottle", 5], ["WineBottle", 5], ["palm", 16], ["plant", 16], ["Plant", 16], ["table", 11], ["Table", 11], ["Defenbakh", 11], ["flower", 16], ["Sofa", 18], ["screen", 20], ["monitor", 20], ["Chair", 9], ["chair", 9], ["Tablesofa", 11]],
    poses_file=Path("/media/david/datasets/ScanNet/house_6/run.txt")
)