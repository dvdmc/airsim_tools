from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, List, Literal, Optional
from PIL import Image
from tqdm import tqdm
import airsim  # type: ignore

import numpy as np
from airsim_tools.depth_conversion import depth_conversion
from airsim_tools.semantics import get_airsim_labels, rgb2label

from poses_tools.frame_converter import FrameConverter  # type: ignore

from airsim_tools.trajectory_functions import *
from airsim_tools.data_saver import NerfstudioDataSave, SaveData, ScanNetDataSave

AirsimSensorDataTypes = Literal["rgb", "depth", "semantic", "pose"]
"""
    List of sensor data to query.
    - "pose": query poses.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
"""

@dataclass
class CameraParams:
    """
    Parameters of the camera.
    """

    width: int
    height: int
    cx: float
    cy: float
    fx: float
    fy: float
    k1: float = 0
    k2: float = 0
    p1: float = 0
    p2: float = 0
    k3: float = 0
    k4: float = 0

    def to_dict(self):
        """
        Returns the parameters as a dictionary.
        """
        return {
            "width": self.width,
            "height": self.height,
            "cx": self.cx,
            "cy": self.cy,
            "fx": self.fx,
            "fy": self.fy,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "k3": self.k3,
            "k4": self.k4,
        }


@dataclass
class AirsimSaverConfig:
    """
    Configuration for the Saver.
    """

    save_dir: Path = Path(".")
    # Path to the directory where the dataset will be saved.

    name: Path = Path("dataset")
    # Name of the dataset.

    mode: Literal["live", "from_poses", "from_function"] = "live"
    # Mode to save the dataset.
    # - "live": save the dataset while manually controlling the camera.
    # - "from_poses": save the dataset from a list of poses.
    # - "from_function": save the dataset from a function that generates poses.

    function: Optional[str] = "spherical"
    # Function to generate the poses.
    # - "spherical": generate poses in a sphere.
    # - "circle": generate poses in a circle.
    # - "line": generate poses in a line.

    function_parameters: dict = field(default_factory=dict)
    # Parameters for the function.

    cameras: List[str] = field(default_factory=list, metadata={"default": ["0"]})
    # List of cameras to save.

    data_types: List[AirsimSensorDataTypes] = field(
        default_factory=list, metadata={"default": ["poses", "rgb", "depth", "semantic"]}
    )
    # List of data_types to save.
    # - "poses": save the poses.
    # - "rgb": save the rgb images.
    # - "depth": save the depth images.
    # - "semantic": save the semantic images.

    origin_transform: Optional[np.ndarray] = None
    # Initial offset of the poses in AirSim coordinates. It indicates the map origin.

    orientation_transform: Optional[float] = None
    # Initial orientation of the poses in AirSim coordinates. It indicates the map orientation.

    save_format: Literal["scannet", "nerfstudio"] = "nerfstudio"
    # Format to save the data and frame of reference for the poses.
    # - "ros": save the data in the format used by ROS.
    # - "nerfstudio": save the data in the format used by nerfstudio.

    semantic_map: Optional[Path] = None
    # Path to the semantic map.

    semantic_config: List[List] = field(default_factory=list, metadata={"default": []})
    # Configuration for the semantic labels.
    # The tuple should contain the substring of the objects to configure
    # and the label to assign to them.

    poses_file: Optional[Path] = None
    # Path to the file where the poses will loaded in the case of "from_poses" mode.

class AirsimSaver:
    """
    Saves data from AirSim.
    """

    def __init__(self, cfg: AirsimSaverConfig):
        self.cfg = cfg

    def _get_function(self):
        """
        Get the function to generate the poses.
        """
        if self.cfg.function == "spherical":
            # Apply directly the named function parameters
            return SphericalPosesFunction(**self.cfg.function_parameters)
            # return DummyPosesFunction()
        else:
            raise NotImplementedError

    def _get_saver(self):
        """
        Get the saver to save the data.
        """
        if self.cfg.save_format == "nerfstudio":
            return NerfstudioDataSave(self.dataset_directory, self.camera_params.to_dict())
        elif self.cfg.save_format == "scannet":
            return ScanNetDataSave(str(self.cfg.name), self.dataset_directory, self.camera_params.to_dict())
        else:
            raise NotImplementedError

    def setup(self):
        """
        Setup the saver.
        """
        self.frame_converter = FrameConverter()
        self.frame_converter.setup_from_yaw(0)
        self.frame_converter.setup_transform_function("airsim", self.cfg.save_format)

        self.client = airsim.VehicleClient()
        self.client.confirmConnection()

        # Sim cfg data
        #######################################################
        # RELEVANT CAMERA DATA (TODO: Adjust to the camera used)
        self.width = 512
        self.height = 512
        self.fov_h = 54.4
        self.cx = float(self.width) / 2
        self.cy = float(self.height) / 2
        fov_h_rad = self.fov_h * np.pi / 180.0
        self.fx = self.cx / (np.tan(fov_h_rad / 2))
        self.fy = self.fx * self.height / self.width
        self.client.simSetFocusAperture(7.0, "0")  # Avoids depth of field blur
        self.client.simSetFocusDistance(100.0, "0")  # Avoids depth of field blur
        self.camera_params = CameraParams(self.width, self.height, self.cx, self.cy, self.fx, self.fy)
        #######################################################
        
        # Set initial position
        if self.cfg.origin_transform is None:
            self.cfg.origin_transform = np.array([0, 0, 0])
        if self.cfg.orientation_transform is None:
            self.cfg.orientation_transform = 0

        # Set the data to query
        self.query_data = []
        if "rgb" in self.cfg.data_types:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.Scene, False, False))
        if "depth" in self.cfg.data_types:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False))
        if "semantic" in self.cfg.data_types:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False))
        if "semantic" in self.cfg.data_types:
            self.setup_semantic_config()

        ### Configure the controller and saver
        if self.cfg.mode == "live":
            # self.save = self.go_live
            raise NotImplementedError
        elif self.cfg.mode == "from_poses":
            if self.cfg.poses_file is None:
                raise ValueError("Please provide a path to the poses file.")
            with open(self.cfg.poses_file, "rb") as f:
                self.poses = []
                for line in f.readlines():
                    line = line.split()
                    self.poses.append(line)
            # Assign the method to save the data
            self.save: Callable = self.save_dataset_poses

        elif self.cfg.mode == "from_function":
            # Assign the function to generate poses
            self.function: BasePosesFunction = self._get_function()
            # Assign the method to save the data
            self.save: Callable = self.save_dataset_function

        # Setup the saver
        self.dataset_directory = self.cfg.save_dir / self.cfg.name
        self.saver = self._get_saver()

    def setup_semantic_config(self):
        # Set all objects in the scene to label 0 in the beggining
        if self.cfg.semantic_config is not None:
            # Set everything to ID 0 using regular expression
            success = self.client.simSetSegmentationObjectID(".*", 0, True)

            # To change the remaining we use the semantic config.
            # For each label, we will create a regular expression 
            # that matches all the objects containing the label as a substring
            regexes = {}
            for label, label_id in self.cfg.semantic_config:
                if label not in regexes:
                    regexes[label] = ".*" + label + ".*"
                else:
                    regexes[label] += "|.*" + label + ".*"
            print("Setting object IDs")
            for label, label_id in tqdm(self.cfg.semantic_config):
                success = self.client.simSetSegmentationObjectID(regexes[label], label_id, True)
                
            print("Finished setting object IDs")

    def process_airsim_data(
        self, position: np.ndarray, orientation: Rotation, responses: List[airsim.ImageResponse]
    ) -> SaveData:
        """
        Process the data from AirSim.
        """
        data = SaveData()
        # Get transform matrix from position-rotation in the specified frame
        translation_ros, rotation_ros = self.frame_converter.airsim_to_ros_pose(position, orientation)
        print("Position: {}".format(position))
        print("Position ros: {}".format(translation_ros))
        print("Orientation: {}".format(orientation.as_matrix()))
        print("Orientation ros: {}".format(rotation_ros.as_matrix()))
        rot_matrix_colmap = rotation_ros.as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix_colmap
        transform_matrix[:3, 3] = translation_ros
        data.pose = transform_matrix

        for response in responses:
            if response.image_type == airsim.ImageType.Scene:
                np_image = (
                    np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    .reshape(response.height, response.width, 3)
                    .copy()
                )
                correct_image = np_image[:, :, ::-1]
                image = Image.fromarray(correct_image)
                data.rgb = image
            elif response.image_type == airsim.ImageType.DepthPerspective:
                img_depth_meters = airsim.list_to_2d_float_array(
                    response.image_data_float, response.width, response.height
                )
                img_depth_meters_corrected = depth_conversion(img_depth_meters, self.fx)
                data.depth = img_depth_meters_corrected

            elif response.image_type == airsim.ImageType.Segmentation:
                # Transform Airsim segmentation image to a different color system
                img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb_airsim = img_data.reshape(response.height, response.width, 3)
                np_rgb_airsim = img_rgb_airsim[:,:,::-1]
                # Get the semantic image
                airsim_colormap = get_airsim_labels()
                semantic = rgb2label(np_rgb_airsim, airsim_colormap)
                data.semantic = semantic

        return data

    def save_data_request(self, idx: int, position: np.ndarray, orientation: Rotation):
        """
        Save the data from a request.
        """
        # Get the pose for airsim
        translation_airsim, rotation_airsim = self.frame_converter.ros_to_airsim_pose(position, orientation)
        quaternion_airsim = rotation_airsim.as_quat()

        airsim_pose = airsim.Pose(
            airsim.Vector3r(x_val=translation_airsim[0], y_val=translation_airsim[1], z_val=translation_airsim[2]),
            airsim.Quaternionr(
                x_val=quaternion_airsim[0],
                y_val=quaternion_airsim[1],
                z_val=quaternion_airsim[2],
                w_val=quaternion_airsim[3],
            ),
        )

        self.client.simSetVehiclePose(airsim_pose, True)

        # Get images
        responses = self.client.simGetImages(self.query_data)

        data = self.process_airsim_data(position, orientation, responses)

        self.saver.save_frame(idx, data)

    def save_dataset_poses(self):
        """
        Save the dataset from poses.
        """
        exposure_adjust = False
        for id, pose_list in enumerate(self.poses):

            # Move to pose
            airsim_pose = airsim.Pose(
                airsim.Vector3r(x_val=float(pose_list[0]), y_val=float(pose_list[1]), z_val=float(pose_list[2])),
                airsim.Quaternionr(
                    x_val=float(pose_list[3]),
                    y_val=float(pose_list[4]),
                    z_val=float(pose_list[5]),
                    w_val=float(pose_list[6]),
                ),
            )

            self.client.simSetVehiclePose(airsim_pose, True)

            # Get images
            responses = self.client.simGetImages(self.query_data)

            data = self.process_airsim_data(
                np.array([float(pose_list[0]), float(pose_list[1]), float(pose_list[2])]),
                Rotation.from_quat([float(pose_list[3]), float(pose_list[4]), float(pose_list[5]), float(pose_list[6])]),
                responses,
            )
            if not exposure_adjust:
                # Wait to adjust exposure
                time.sleep(2)
                exposure_adjust = True
                
            self.saver.save_frame(id, data)

        self.saver.post_setup()

    def save_dataset_function(self):
        """
        Save the dataset from a function.
        """
        positions, orientations = self.function.get_all_poses()

        for i in range(len(positions)):
            # Get the pose for airsim
            translation_airsim, rotation_airsim = self.frame_converter.ros_to_airsim_pose(positions[i], orientations[i])

            # Apply orientation offset (pre-multiplication!) TODO: Add offsets to frame converter
            if self.cfg.orientation_transform is not None:
                orientation_transform = Rotation.from_euler("z", self.cfg.orientation_transform, degrees=True)
            else:
                orientation_transform = Rotation.identity()
            rotation_airsim_yaw_corrected =  rotation_airsim * orientation_transform
            quaternion_airsim = rotation_airsim_yaw_corrected.as_quat()  # Scipy quat uses: xyzw
            translation_airsim_offset_corrected = translation_airsim + self.cfg.origin_transform
            translation_airsim_vector = airsim.Vector3r(
                x_val=translation_airsim_offset_corrected[0], y_val=translation_airsim_offset_corrected[1], z_val=translation_airsim_offset_corrected[2]
            )
            rotation_airsim_quaternion = airsim.Quaternionr(
                x_val=quaternion_airsim[0],
                y_val=quaternion_airsim[1],
                z_val=quaternion_airsim[2],
                w_val=quaternion_airsim[3],
            )
        
            airsim_pose = airsim.Pose(
                translation_airsim_vector,
                rotation_airsim_quaternion,
            )
            self.client.simSetVehiclePose(airsim_pose, True)
            if i == 0:
                # Wait to adjust exposure
                time.sleep(2)

            # time.sleep(0.1)
            # Get images
            responses = self.client.simGetImages(self.query_data)

            data = self.process_airsim_data(positions[i], orientations[i], responses)

            self.saver.save_frame(i, data)
