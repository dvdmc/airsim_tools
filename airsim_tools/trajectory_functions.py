from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

from phd_utils.poses.frame_converter import FrameConverter

class BasePosesFunction:
    def __init__(self) -> None:
        pass

    def setup(self):
        """
            This function is used to parametrize the function
            e.g. set the initial position of the object, radius
            of a sphere, etc.
        """
        raise NotImplementedError

    def get_all_poses(self):
        """
            This function is used to generate the poses.
        """
        raise NotImplementedError
    
    def get_pose(self):
        """
            Optional function to get the pose at a specific index
            depending on the input variables.
        """
        raise NotImplementedError

class DummyPosesFunction(BasePosesFunction):
    """
        This function generates dummy poses for testing
        the basic coordinate changes.
    """
    def __init__(self) -> None:
        pass

    def setup(self):
        pass

    def get_all_poses(self):
        translations = []
        rotations = []

        # Generate identity rotation and movement in x, y and z
        translations.append(np.array([0, 0, 0]))
        rotations.append(Rotation.identity())

        # This moved in the +red axis
        translations.append(np.array([1, 0, 0]))
        rotations.append(Rotation.identity())

        # This moved in the +blue axis
        translations.append(np.array([0, 1, 0]))
        rotations.append(Rotation.identity())

        # This moved in the -green axis
        translations.append(np.array([0, 0, 1]))
        rotations.append(Rotation.identity())

        # This rotated around +red axis
        translations.append(np.array([2, 0, 0]))
        rotations.append(Rotation.from_euler("xyz", [np.pi/2, 0, 0]))

        # This rotated around -green axis
        translations.append(np.array([0, 2, 0]))
        rotations.append(Rotation.from_euler("xyz", [0, np.pi/2, 0]))

        # This rotated around +blue axis
        translations.append(np.array([0, 0, 2]))
        rotations.append(Rotation.from_euler("xyz", [0, 0, np.pi/2]))

        return translations, rotations

    def get_pose(self):
        return np.array([0, 0, 0]), Rotation.from_euler("xyz", [0, 0, 0])
    
class SphericalPosesFunction(BasePosesFunction):
    """
        This function generates poses in a spherical coordinate system.
        The orientation of the pose is always towards the center of the sphere.

        Args:
            radius: radius of the sphere
            theta: angle in the x-y plane
            phi: angle in the x-z plane
            offset: offset of the sphere
            angle_offset: angle offset of the sphere
    """
    def __init__(self, radius: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> None:
        self.radius = radius
        self.theta = theta
        self.phi = phi


    def setup(self, radius: np.ndarray, theta: np.ndarray, phi: np.ndarray):
        """
            This function is used to parametrize the function
            e.g. set the initial position of the object, radius
            of a sphere, etc.

            Args:
                radius: radius of the sphere
                theta: angle in the x-y plane
                phi: angle in the x-z plane
                offset: offset of the sphere
                angle_offset: angle offset of the sphere
        """
        self.radius = radius
        self.theta = theta
        self.phi = phi


    def get_all_poses(self) -> Tuple[List[np.ndarray], List[Rotation]]:
        """
            This function is used to generate all the poses.

            Returns:
                translations: list of translations
                rotations: list of rotations
        """
        translations = []
        rotations = []
        for r in self.radius:
            for phi in self.phi: # angle in x-y plane around z axis
                for theta in self.theta: # angle from z axis to x-y plane
                    theta_rad = np.deg2rad(theta)
                    phi_rad = np.deg2rad(phi)
                    translation, rotation = FrameConverter.spherical_to_cartesian(r, theta_rad, phi_rad)
                    
                    translations.append(translation)
                    rotations.append(rotation)
        return translations, rotations
    
    def get_pose(self, radius: float, theta: float, phi: float) -> Tuple[np.ndarray, Rotation]:
        """
            Optional function to get the pose at a specific index
            depending on the input variables.

            Args:
                radius: radius of the sphere
                theta: angle in the x-y plane
                phi: angle in the x-z plane

            Returns:
                translation: translation
                rotation: rotation
        """
        t, rot = FrameConverter.spherical_to_cartesian(radius, theta, phi)

        return t, rot
