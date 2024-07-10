from typing import List
import tyro
import airsim

def main(position: List[float], size: List[float], resolution: float, output_file: str):
    """
        Generate a voxel grid, centered at position, with size and resolution
        and save it to output_file

        Args:
            position: center of the voxel grid
            size: size of the voxel grid
            resolution: resolution of the voxel grid
            output_file: output file
    """
    # Create client
    client = airsim.VehicleClient()
    client.confirmConnection()

    # Create voxel grid Coordinates are x front, y up, z left
    position_v = airsim.Vector3r(position[0], position[1], position[2])
    client.simCreatePCL(position_v, x=int(size[0]), y=int(size[1]), z=int(size[2]), res=resolution, of=output_file)

if __name__ == "__main__":
    tyro.cli(main)