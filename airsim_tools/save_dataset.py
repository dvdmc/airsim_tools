from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, List, Literal, Optional
from PIL import Image
import airsim

import numpy as np
from airsim_tools.depth_conversion import depth_conversion
from airsim_tools.semantics import airsim2class_id

from poses_tools.frame_converter import FrameConverter

from airsim_tools.trajectory_functions import *
from airsim_tools.data_saver import NerfstudioDataSave, SaveData


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

    sensors: List[Literal["poses", "rgb", "depth", "semantic", "lidar"]] = field(
        default_factory=list, metadata={"default": ["poses", "rgb", "depth", "semantic"]}
    )
    # List of sensors to save.
    # - "poses": save the poses.
    # - "rgb": save the rgb images.
    # - "depth": save the depth images.
    # - "semantic": save the semantic images.

    origin_transform: Optional[np.ndarray] = None
    # Initial offset of the poses in AirSim coordinates. It indicates the map origin.

    orientation_transform: Optional[float] = None
    # Initial orientation of the poses in AirSim coordinates. It indicates the map orientation.

    save_format: Literal["ros", "nerfstudio"] = "nerfstudio"
    # Format to save the data and frame of reference for the poses.
    # - "ros": save the data in the format used by ROS.
    # - "nerfstudio": save the data in the format used by nerfstudio.

    semantic_map: Optional[Path] = None
    # Path to the semantic map.

    semantic_config: Optional[List[Tuple[str, int]]] = None
    # Configuration for the semantic labels.
    # The tuple should contain the substring of the objects to configure
    # and the label to assign to them.


class AirsimSaver:
    """
    Saves data from AirSim.
    """

    def __init__(self, config: AirsimSaverConfig):
        self.config = config

    def _get_function(self):
        """
        Get the function to generate the poses.
        """
        if self.config.function == "spherical":
            # Apply directly the named function parameters
            return SphericalPosesFunction(**self.config.function_parameters)
            # return DummyPosesFunction()
        else:
            raise NotImplementedError

    def _get_saver(self):
        """
        Get the saver to save the data.
        """
        if self.config.save_format == "nerfstudio":
            return NerfstudioDataSave(self.dataset_directory, self.camera_params.to_dict())
        else:
            raise NotImplementedError

    def setup(self):
        """
        Setup the saver.
        """
        self.frame_converter = FrameConverter()
        self.frame_converter.setup_from_yaw(0)
        self.frame_converter.setup_transform_function("airsim", self.config.save_format)

        self.client = airsim.VehicleClient()
        self.client.confirmConnection()

        # Sim config data
        #######################################################
        # RELEVANT CAMERA DATA
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
        if self.config.origin_transform is None:
            self.config.origin_transform = np.array([0, 0, 0])
        if self.config.orientation_transform is None:
            self.config.orientation_transform = 0

        # Set the data to query
        self.query_data = []
        if "rgb" in self.config.sensors:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.Scene, False, False))
        if "depth" in self.config.sensors:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False))
        if "semantic" in self.config.sensors:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False))

        if "semantic" in self.config.sensors:
            self.setup_semantic_config()

        ### Configure the controller and saver
        if self.config.mode == "live":
            # self.save = self.go_live
            raise NotImplementedError
        elif self.config.mode == "from_poses":
            # self.save = self.save_dataset_poses
            raise NotImplementedError
        elif self.config.mode == "from_function":
            # Assign the function to generate poses
            self.function: BasePosesFunction = self._get_function()
            # Assign the mehtod to save the data
            self.save: Callable = self.save_dataset_function

        # Setup the saver
        self.dataset_directory = self.config.save_dir / self.config.name
        self.saver = self._get_saver()

    def setup_semantic_config(self):
        # Set all objects in the scene to label 0 in the beggining
        if self.config.semantic_config is not None:
            for object_id in self.client.simListSceneObjects(): #type: ignore
                changed = False
                for object_str, label in self.config.semantic_config:
                    if object_str in object_id:
                        changed = True
                        success = self.client.simSetSegmentationObjectID(object_id, label)
                        if not success:
                            print("Could not set segmentation object ID for {}".format(object_id))
                        else:
                            print("Changed object ID to {} for {}".format(label, object_id))
                        break
                if not changed:  # TODO: Check if this is faster than just setting all objects to 0
                    self.client.simSetSegmentationObjectID(object_id, 0)
            print("Finished setting object IDs")

    def process_airsim_data(
        self, position: np.ndarray, orientation: Rotation, responses: List[airsim.ImageResponse]
    ) -> SaveData:
        """
        Process the data from AirSim.
        """
        data = SaveData()
        # Get transform matrix from position-rotation in the specified frame
        translation_colmap, rotation_colmap = self.frame_converter.ros_to_nerfstudio_pose(position, orientation)
        print("Position: {}".format(position))
        print("Position colmap: {}".format(translation_colmap))
        print("Orientation: {}".format(orientation.as_matrix()))
        print("Orientation colmap: {}".format(rotation_colmap.as_matrix()))
        rot_matrix_colmap = rotation_colmap.as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix_colmap
        transform_matrix[:3, 3] = translation_colmap
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
                semantic = airsim2class_id(np_rgb_airsim)
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

    def save_dataset_function(self):
        """
        Save the dataset from a function.
        """
        positions, orientations = self.function.get_all_poses()

        for i in range(len(positions)):
            # Get the pose for airsim
            translation_airsim, rotation_airsim = self.frame_converter.ros_to_airsim_pose(positions[i], orientations[i])

            # Apply orientation offset (pre-multiplication!) TODO: Add offsets to frame converter
            if self.config.orientation_transform is not None:
                orientation_transform = Rotation.from_euler("z", self.config.orientation_transform, degrees=True)
            else:
                orientation_transform = Rotation.identity()
            rotation_airsim_yaw_corrected =  rotation_airsim * orientation_transform
            quaternion_airsim = rotation_airsim_yaw_corrected.as_quat()  # Scipy quat uses: xyzw
            translation_airsim_offset_corrected = translation_airsim + self.config.origin_transform
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
