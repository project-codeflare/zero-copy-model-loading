#! /bin/bash


# Create conda environment to run the notebooks in this directory.
#
# By default, the environment will be located in the directory "env"
# immediately under this one. To override that setting,
# pass the subdirectory name as the first argument to this script, i.e.
#
# $ ./env.sh my_dir_name

PYTHON_VERSION=3.8

############################
# HACK ALERT *** HACK ALERT
# The friendly folks at Anaconda thought it would be a good idea to make the
# "conda" command a shell function.
# See https://github.com/conda/conda/issues/7126
# The following workaround will probably be fragile.
if [ -z "$CONDA_HOME" ]
then
    echo "Error: CONDA_HOME not set."
    exit
fi
if [ -e "${CONDA_HOME}/etc/profile.d/conda.sh" ]
then
    # shellcheck disable=SC1090
    . "${CONDA_HOME}/etc/profile.d/conda.sh"
else
    echo "Error: CONDA_HOME (${CONDA_HOME}) does not appear to be set up."
    exit
fi
# END HACK
############################

# Check whether the user specified an environment name.
if [ "$1" != "" ]; then
    ENV_DIR=$1
else
    ENV_DIR="env"
fi
echo "Creating an Anaconda environment at ./${ENV_DIR}"


# Remove the detrius of any previous runs of this script
rm -rf ./${ENV_DIR}

# Note how we explicitly install pip on the line that follows. THIS IS VERY
# IMPORTANT!
conda create -y -p ${ENV_DIR} python=${PYTHON_VERSION} pip
conda activate ./${ENV_DIR}

################################################################################
# Install packages with conda

# We currently install JupyterLab from conda because the pip packages are 
# broken for Anaconda environments with Python 3.6 and 3.8 on Mac, as of
# April 2021.
conda install -y -c conda-forge jupyterlab
conda install -y -c conda-forge/label/main nodejs
conda install -y -c conda-forge jupyterlab-git

################################################################################
# Install packages with pip

# Pip dependencies are all in requirements.txt
pip install -r requirements.txt

# Install the local source tree in editable mode
 pip install --editable .

################################################################################
# Custom install steps

# Elyra extensions to JupyterLab (enables git integration, debugger, workflow
# editor, outlines, and other features)
pip install --upgrade --use-deprecated=legacy-resolver elyra

# Rebuild JupyterLab environment 
jupyter lab build

jupyter --version
echo " "
jupyter serverextension list
echo " "
jupyter labextension list
echo " "

conda deactivate

echo "Anaconda environment at ./${ENV_DIR} successfully created."
echo "To use, type 'conda activate ./${ENV_DIR}'."

