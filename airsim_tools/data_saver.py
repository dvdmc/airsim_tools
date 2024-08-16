"""
    This file contains the code to save data to different formats.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Optional

import numpy as np
from PIL import Image

from airsim_tools.semantics import get_color_map, class_to_rgb

@dataclass
class SaveData:
    pose: np.ndarray = np.eye(4)
    """ Pose of the camera. """

    rgb: Optional[Image.Image] = None
    """ RGB image. """

    depth: Optional[np.ndarray] = None
    """ Depth image. Input an array to allow further transformations."""

    semantic: Optional[np.ndarray] = None
    """ Semantic image. Input an array to allow further transformations."""
    
    lidar: Optional[np.ndarray] = None
    """ Lidar point cloud. """


class BaseDataSave:
    """
    Base class for saving data.
    """

    def __init__(self, save_dir: Path):
        # Check if save_dir exists.
        if not save_dir.exists():
            print(f"Save directory '{save_dir}' does not exist. Creating it.")
            save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

    def save_frame(self, index: int, data: SaveData):
        """
        Saves a frame.
        """
        raise NotImplementedError("Save frame not implemented for this saver.")
    
    def post_setup(self):
        """
        Performs any post setup.
        """
        raise NotImplementedError("Post setup not implemented for this saver.")

class ScanNetDataSave(BaseDataSave):
    """
    Saves data in the format used by ScanNet.
    The structure is:
    - color: 
        %d.jpg
    - depth: 
        %d.png (16bit in millimeters)
    - intrinsic:
        extrinsic_color.txt (identity)
        extrinsic_depth.txt (identity)
        intrinsic_color.txt
        intrinsic_depth.txt 
    - label: 
        %d.png (numbers)
    - label_color
        %d.png (color maps)
    - pose
        %d.txt
    - {scene_name}.txt:
        axisAlignment = 0.258819 0.965926 0.000000 -3.908060 -0.965926 0.258819 0.000000 2.228620 0.000000 0.000000 1.000000 -0.078392 0.000000 0.000000 0.000000 1.000000 
        colorHeight = 968
        colorWidth = 1296
        depthHeight = 480
        depthWidth = 640
        fx_color = 1170.187988
        fx_depth = 578.000000
        fy_color = 1170.187988
        fy_depth = 578.000000
        mx_color = 647.750000
        mx_depth = 319.500000
        my_color = 483.750000
        my_depth = 239.500000
        numColorFrames = 2949
        numDepthFrames = 2948
        numIMUmeasurements = 6305
        sceneType = Office
    """
    def __init__(self, scene_name: str, save_dir: Path, camera_params: dict, sensors: List[str] = ["rgb", "depth", "semantic"], color_map_name: str = "coco_voc"):
        super().__init__(save_dir)
        self.camera_params = camera_params
        self.color_map = get_color_map(color_map_name, bgr=False)
        self.scene_name = scene_name
        self.sensors = sensors
        self._setup()
    
    def post_setup(self):
        self.scene_file = self.save_dir / f"{self.scene_name}.txt"
        scene_config = f"""colorHeight = {self.camera_params["height"]}
colorWidth = {self.camera_params["width"]}
depthHeight = {self.camera_params["height"]}
depthWidth = {self.camera_params["width"]}
fx_color = {self.camera_params["fx"]}
fx_depth = {self.camera_params["fx"]}
fy_color = {self.camera_params["fy"]}
fy_depth = {self.camera_params["fy"]}
mx_color = {self.camera_params["cx"]}
mx_depth = {self.camera_params["cx"]}
my_color = {self.camera_params["cy"]}
my_depth = {self.camera_params["cy"]}
numColorFrames = {self.num_color}
numDepthFrames = {self.num_depth}
numIMUmeasurements = 0
sceneType = "Office"
"""
        with open(self.scene_file, "w") as f:
            f.write(scene_config)

    def _setup(self):
        """
        Setup the saver.
        """
        # Create the parent directory.
        print(f"Saving data to '{self.save_dir}'")
        if "rgb" in self.sensors:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.rgb_dir = self.save_dir / "color"
            self.rgb_dir.mkdir(parents=True, exist_ok=True)
        if "depth" in self.sensors:
            self.depth_dir = self.save_dir / "depth"
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        if "semantic" in self.sensors:
            self.semantic_dir = self.save_dir / "label"
            self.semantic_dir.mkdir(parents=True, exist_ok=True)
            self.semantic_color_dir = self.save_dir / "label_color"
            self.semantic_color_dir.mkdir(parents=True, exist_ok=True)

        self.poses_dir = self.save_dir / "pose"
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        self.intrinsics_dir = self.save_dir / "intrinsic"
        self.intrinsics_dir.mkdir(parents=True, exist_ok=True)

        # Create intrinsic files
        intrinsic = np.eye(3)
        intrinsic[0, 0] = self.camera_params["fx"]
        intrinsic[1, 1] = self.camera_params["fy"]
        intrinsic[0, 2] = self.camera_params["cx"]
        intrinsic[1, 2] = self.camera_params["cy"]
        intrinsic_path = self.save_dir / "intrinsic" / "intrinsic_color.txt"
        np.savetxt(intrinsic_path, intrinsic, fmt='%.6f')

        intrinsic[0, 0] = self.camera_params["fx"]
        intrinsic[1, 1] = self.camera_params["fy"]
        intrinsic[0, 2] = self.camera_params["cx"]
        intrinsic[1, 2] = self.camera_params["cy"]
        intrinsic_path = self.save_dir / "intrinsic" / "intrinsic_depth.txt"
        np.savetxt(intrinsic_path, intrinsic, fmt='%.6f')

        # Create extrinsic files
        extrinsic = np.eye(4)
        extrinsic_color_path = self.save_dir / "intrinsic" / "extrinsic_color.txt"
        extrinsic_depth_path = self.save_dir / "intrinsic" / "extrinsic_depth.txt"
        np.savetxt(extrinsic_color_path, extrinsic, fmt='%.6f')
        np.savetxt(extrinsic_depth_path, extrinsic, fmt='%.6f')

        self.num_color = 0
        self.num_depth = 0

    def save_frame(self, index: int, data: SaveData):
        """
        Saves a frame.
        """
        if data is not None:
            if "rgb" in self.sensors and data.rgb is not None:
                file_name = f"{index}.jpg"
                data.rgb.save(self.rgb_dir / file_name)
                self.num_color += 1
            if "depth" in self.sensors and data.depth is not None:
                file_name = f"{index}.png"
                depth_img_in_millimeters = data.depth * 1000
                depth_16bit = np.clip(depth_img_in_millimeters, 0, 65535)
                depth_img = Image.fromarray(depth_16bit.astype("uint16"))
                depth_img.save(self.depth_dir / file_name)
                self.num_depth += 1
            if "semantic" in self.sensors and data.semantic is not None:
                file_name = f"{index}.png"
                semantic_img = Image.fromarray(data.semantic.astype("uint8"))
                semantic_img.save(self.semantic_dir / file_name)
                label_color = Image.fromarray(class_to_rgb(data.semantic, self.color_map), mode="RGB")
                label_color.save(self.semantic_color_dir / file_name)

            # The pose is required!
            file_name = f"{index}.txt"
            np.savetxt(self.poses_dir / file_name, data.pose)

class NerfstudioDataSave(BaseDataSave):
    """
    Saves data in the format used by Nerfstudio.
    This is:
    - RGB files are saved as:  frame_00071.png
    - Depth files are saved as: frame_00071.png
    - Camera parameters are saved in a transforms.json file as:
        {
            "frames": [
                {
                    "fl_x": 1826.620849609375,
                    "fl_y": 1826.0267333984375,
                    "k1": 0,
                    "k2": 0,
                    "k3": 0,
                    "k4": 0,
                    "p1": 0,
                    "p2": 0,
                    "cx": 540.0,
                    "cy": 960.0,
                    "w": 1080,
                    "h": 1920,
                    "aabb_scale": 16,
                    "file_path": "sensor0\\rgb\\000001.png",
                    "depth_file_path": "sensor0\\depth\\000001.png",
                    "transform_matrix": [
                        [
                            0.024848394095897675,
                            -0.3371071219444275,
                            0.9411383271217346,
                            1.4621940851211548
                        ],
                        [
                            0.9996904134750366,
                            0.009575003758072853,
                            -0.022964637726545334,
                            -0.03897222876548767
                        ],
                        [
                            -0.0012698600767180324,
                            0.9414176344871521,
                            0.33724066615104675,
                            1.6487826108932495
                        ],
                        [
                            0.0,
                            -0.0,
                            0.0,
                            1.0
                        ]
            ],
            ...
            ]
        }
    """
    def __init__(self, save_dir: Path, camera_params: dict, sensors: List[str] = ["rgb", "depth", "semantic"]):
        super().__init__(save_dir)
        self.camera_params = camera_params
        self.sensors = sensors
        self._setup()

    def _setup(self):
        """
        Setup the saver.
        """
        # Create the parent directory.
        print(f"Saving data to '{self.save_dir}'")
        if "rgb" in self.sensors:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.rgb_dir = self.save_dir / "images" / "rgb"
            self.rgb_dir.mkdir(parents=True, exist_ok=True)
        if "depth" in self.sensors:
            self.depth_dir = self.save_dir / "images" / "depth"
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        if "semantic" in self.sensors:
            self.semantic_dir = self.save_dir / "images" / "label"
            self.semantic_dir.mkdir(parents=True, exist_ok=True)

        self.transforms_file = self.save_dir / "transforms.json"
        # Check if the transforms file exists.
        if not self.transforms_file.exists():
            print(f"Transforms file '{self.transforms_file}' does not exist. Creating it.")
            with open(self.transforms_file, "w") as f:
                transforms = {"frames": []}
                json.dump(transforms, f, indent=4)
        else :
            print(f"Transforms file '{self.transforms_file}' already exists. Cleaning it.")
            with open(self.transforms_file, "w") as f:
                transforms = {"frames": []}
                json.dump(transforms, f, indent=4)

    def save_frame(self, index: int, data: SaveData):
        """
        Saves a frame.
        """
        if data is not None:
            frame = {}
            frame["fl_x"] = self.camera_params["fx"]
            frame["fl_y"] = self.camera_params["fy"]
            frame["k1"] = self.camera_params["k1"]
            frame["k2"] = self.camera_params["k2"]
            if "k3" in self.camera_params:
                frame["k3"] = self.camera_params["k3"]
            if "k4" in self.camera_params:
                frame["k4"] = self.camera_params["k4"]

            frame["p1"] = self.camera_params["p1"]
            frame["p2"] = self.camera_params["p2"]
            frame["cx"] = self.camera_params["cx"]
            frame["cy"] = self.camera_params["cy"]
            frame["w"] = self.camera_params["width"]
            frame["h"] = self.camera_params["height"]
            frame["aabb_scale"] = 1.0
            file_name = f"frame_{index:06d}.png"
            if data.rgb is not None:
                frame["file_path"] = f"{self.rgb_dir / file_name}"
            if data.depth is not None:
                frame["depth_file_path"] = f"{self.depth_dir / file_name}"

            # The pose is required!
            frame["transform_matrix"] = data.pose.tolist()

            # Open the transforms file as json and append the frame to the frames list.
            with open(self.transforms_file, "r") as f:
                print(f"Saving frame {index} to '{self.transforms_file}'")
                transforms = json.load(f)
                transforms["frames"].append(frame)
            with open(self.transforms_file, "w") as f:
                json.dump(transforms, f, indent=4)

        # Save the rest of the data.
        if data.rgb is not None:
            self._save_rgb(index, data.rgb)
        if data.depth is not None:
            self._save_depth(index, data.depth)
        if data.semantic is not None:
            self._save_semantic(index, data.semantic)

    def _save_rgb(self, index: int, rgb: Image.Image):
        """
        Save the rgb image.
        """
        file_name = f"frame_{index:06d}.png"
        file_path = self.rgb_dir / file_name
        rgb.save(file_path)

    def _save_depth(self, index: int, depth: np.ndarray):
        """
        Save the depth image.
        """
        # Convert depth_img to millimeters to fill out 16bit unsigned int space (0..65535).
        # Also clamp large values (e.g. SkyDome) to 65535
        depth_img_in_millimeters = depth * 1000
        depth_16bit = np.clip(depth_img_in_millimeters, 0, 65535)
        depth_img = Image.fromarray(depth_16bit.astype("uint16"))
        file_name = f"frame_{index:06d}.png"
        file_path = self.depth_dir / file_name
        depth_img.save(file_path)

    def _save_semantic(self, index: int, semantic: np.ndarray):
        """
        Save the semantic image.
        """
        # Save in label format (iMap):
        file_name = f"frame_{index:06d}.png"
        file_path = self.semantic_dir / file_name
        semantic_img = Image.fromarray(semantic.astype("uint8"))
        semantic_img.save(file_path)
        # Save in color format:
        # img_rgb_colormap = label2rgb(
        #             semantic, self.color_map, self.n_classes)
        # img_rgb_colormap = img_rgb_colormap.fromarray(img_rgb_colormap)
        # img_rgb_colormap.save(file_path)
