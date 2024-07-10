# Airsim tools

This repository contains certain tools related to interfacing the Airsim simulator
with outside tools. Currently, the main functionality is to save datasets based on different
acquisition methods and other functionalities used for data and pose conversion, and semantic tools.

## Requirements

The requirements are specified in the project file. Other than standard Python libraries, the tools
use the poses_tools.

## Installation

Clone the repository with:

```
git clone https://github.com/dvdmc/airsim_tools
```

It is highly recommended to use a virtual environment. Some of the Airsim dependencies are incompatible with newer modules.
Probably, you already have Airsim and its Python library installed. Otherwise, follow the installation
process in this [link](https://microsoft.github.io/AirSim/apis/#python-quickstart). 
The package can be installed using the pyproject file:

```
pip install . # use -e for installing an editable version in case you want to modify / debug
```

## Structure

The entry points are in the `scripts` folder. As for now, the code mainly includes the `DataSaver` class
intended to save datasets from the Airsim simulator using the computer vision mode. There are different
scripts to correct the depth and adapt semantics. The scripts can be configured using the scripts in
the `config` folder. Aside from the scripts, there are tools for dealing with data and pose conversion and semantic tools.
