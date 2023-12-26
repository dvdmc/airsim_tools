"""
    This file contains the code to save data to different formats.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Optional

import numpy as np
from PIL import Image
import airsim


@dataclass
class SaveData:
    pose: Optional[np.ndarray] = None
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
        raise NotImplementedError


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
