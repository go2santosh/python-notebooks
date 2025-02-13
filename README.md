# python-notebooks

This git repository contains experimental python notebooks. 

## Use of Virtual Environments

Many of the experimental python notebooks are contained in a dedicated subfolder along with requirements.txt file. 

I use VSCode for development. Follow these steps in VSCode to re-create virtual environment using requirements.txt file.

Steps:
1. Open the python notebook folder in Visual Studio Code.
2. Use the Python: Create Environment command to create a new environment and install dependencies from requirements.txt.
3. This will create a new environment and install all the dependencies listed in the requirements.txt file.

You may install additional dependencies during experimentations. You can update the requirements.txt file using the following command in the terminal:

pip freeze > requirements.txt

This will update the requirements.txt file with all the installed packages and their versions.

## Saving Disk Space

Each subfolder with its own virtual environment consumes a significant amount of disk storage. If you need to free disk space then consider deleting .venv folders.