# Python Setup Guide
We use python 3 for programming assignments in this course. We don't expect the specific version of python 3 to make a difference - anything 3.6 or later should work. This guide is intended to help you set up python on your local mahcine, if you haven't already. You don't have to follow this guide if you have other preferred ways to use python.

## Installation
### Windows
Windows systems don't have python pre-installed, we recommend installing the Anaconda distribution. A few other options are listed, but we won't be providing support for those methods.
1. [Anaconda](https://www.anaconda.com/distribution/): Follow the link, download and install the appropriate version. The installed ***Powershell Prompt*** is the command line environment you want to use
2. Install python following this [documentation](https://docs.python.org/3/using/windows.html)
3. Install the Ubuntu 18.04 distribution of [Windows subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and use the Ubuntu Bash terminal.
4. Set up a Linux virtual machine
5. Install a Linux OS alongside your current OS as a dual boot.
6. Install an integrated development environment (IDE) that can manage dependencies.

### MacOS
In your terminal, run `python3 --version` to see if python is already installed, and its version if installed. <!-- check this -->
1. [Anaconda](https://www.anaconda.com/distribution/): Follow the link, download and install the appropriate version. The installed ***Powershell Prompt*** is the command line environment you want to use <!-- check this -->
2. Install [Brew](https://brew.sh/) then run `brew install python3` in your terminal

### Linux
Python 3 is likely installed, run `python3 --version` to check and see its version. If not, run in your terminal:
1. `sudo apt-get update`
2. `sudo apt-get install python3`
3. `sudo apt-get install python3-pip`

## Virtual Environment
We recommend maintaining a virtual environment for this course to separate the dependencies from your other python projects, though it's not required. When you are in a virtual environment, your command line path is prefixed by the environment name in parenthesis.

### Anaconda
- Run `conda create -n <environment name>` to create an environment
- Run `conda activate <environment name>` to activate an environment
- Run `conda deactivate` to deactivate the current environment

### virtualenv and venv on Linux and MacOS
You might need to use brew/apt-get to install virtualenv or venv
- Creating an environment, use one of the two:
    - `virtualenv -p python3 <environment name>`
    - `python3 -m venv <environment name>`
- Activating an environment: `source <environment name>/bin/activate`
- Deactivating current environment: `deactivate`