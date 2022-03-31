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

Usage()
{
    echo "Usage ./env.sh [-d dir_name] [-p] [-h]"
    echo "Where:"
    echo "  -d dir_name specifies the location of the environment."
    echo "     (default is ./env)"
    echo "  -p means to install the zerocopy module from PyPI"
    echo "     (default is to install from local source)"
    echo "  -h prints this message" 
}

INSTALL_FROM_PYPI=false
ENV_DIR="env"

while getopts ":hpd:" option; do
    case $option in
        h) # display Help
            Usage
            exit;;
        d) # Specify directory
            ENV_DIR=$OPTARG;;
        p) # Install from PyPI
            INSTALL_FROM_PYPI=true;;
        \?) # Invalid option
            Usage
            exit;;
    esac
done

echo "Creating an Anaconda environment at ./${ENV_DIR}"
if [ "$INSTALL_FROM_PYPI" = true ] ; then
    echo "Will install zerocopy package from PyPI"
else
    echo "Will install zerocopy package from local source tree"
fi

# Remove the detritus of any previous runs of this script
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

if [ "$INSTALL_FROM_PYPI" = true ] ; then
    pip install zerocopy
else
    # Install the local source tree for the `zerocopy` package in editable mode
    pip install --editable ./package
fi

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

