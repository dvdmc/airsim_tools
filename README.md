# Airsim tools

This repository contains certain tools related to interfacing the Airsim simulator
with outside tools. Currently, the main functionality is to save datasets based on different
acquisition methods.

## Requirements

The requirements are specified in the project file. Other than standard Python libraries, the tools
use the poses_tools.

## Installation

It is highly recommended to use a virtual environment. Some of the Airsim dependencies are incompatible with newer modules:

```
conda create -n airsim
```

Probably, you already have Airsim and its Python library installed. Otherwise, follow the installation
process in this [link](https://microsoft.github.io/AirSim/apis/#python-quickstart). Then, you can install all the dependencies with the following command from this folder. The package will be installed using the pyproject file:

```
pip install -e .
```

## Structure

The entrypoints are in the `scripts` folder. As for now, the code mainly includes the `DataSaver` class
intended to save datasets from the Airsim simulator using the computer vision mode. There are different
scripts to correct the depth and adapt semantics. The scripts can be configured using the scripts in
the `config` folder.

## TODO

- Include a simplified version of the ROS bridge.

- Different planners to interface with Airsim.