# zero-copy-model-loading

Code to accompany the Medium post, ["How to Load PyTorch Models 340 Times Faster
with
Ray"](https://medium.com/ibm-data-ai/how-to-load-pytorch-models-340-times-faster-with-ray-8be751a6944c).

## Notebooks

Notebooks can be found in the `notebooks` directory:
* `zero_copy_loading.ipynb`: The notebook that was used when authoring the 
  original blog post.
* `benchmark/benchmark.ipynb`: The notebook that was used when authoring the
  second post in the series.

Instructions to run notebooks:
1. Install `bash` and either `anaconda` or `miniconda`.
1. Check out a copy of this repository and navigate to the root directory of
   your local copy.
1. Run the script `./env.sh`, which creates an Anaconda environment under 
   `<root of your local copy>/env`.
   ```
   ./env.sh
   ```
1. Activate the Anaconda environment:
   ```
   conda activate ./env
   ```
1. Run JupyterLab:
   ```
   jupyter lab
   ```
1. Navigate to the `notebooks` directory and open up the Jupyter notebook of your choice.


## Python Package

This repository also contains the source code for the `zerocopy` library.
`zerocopy` is a Python package that provides functions for implementing
zero-copy model loading of PyTorch models on Ray.

You can find the source code for the package inside the `package` directory.

