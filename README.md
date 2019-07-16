
**Online Notebooks:**

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lento234/python-tutorial/master?urlpath=lab)
[![Azure](https://notebooks.azure.com/launch.svg)](https://notebooks.azure.com/import/gh/lento234/python-tutorial)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lento234/python-tutorial/blob/master/04_numerical_python/04_numpy.ipynb)

**Continuous integration:**

[![Build Status](https://travis-ci.com/lento234/python-tutorial.svg?branch=master)](https://travis-ci.com/lento234/python-tutorial)

# Python Workshop

Sources (derived and modified from):
- [Scipy 2019 jupyterlab tutorial](https://github.com/jupyterlab/scipy2019-jupyterlab-tutorial)
- [Lecture series on scientific python by jrjohansson](https://github.com/jrjohansson/scientific-python-lectures)
- [Scipy Lecture notes](https://scipy-lectures.org/)
- [Quantitative Big Imaging 2019 by Kevin Mader (ETHZ)](https://github.com/kmader/Quantitative-Big-Imaging-2019)
- [Dask examples tutorial](https://github.com/dask/dask-examples)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

### The goals of 1st workshop are: 

- basic overview of python
- substituting your matlab workflow with python
- learning to use excel, csv files in python, and
- version controlling and backup up your data / code.


### Detailed content of the workshop:

1. Installation using anaconda
2. Python environments with conda
3. Working with python in terminal (ipython), IDE (spyderlib), notebooks (jupyterlab)
4. Basic python
5. Numerical computation (i.e., Matlab-esq) programming in python (numpy)
6. Matlab-esq plotting in python (matplotlib)
7. Reading, writing (excel, csv, ...) and statistical data analysis (pandas)
8. Basic intro to git (more advanced later)


## Software installation

1. Install the full [anaconda
   distribution](https://www.anaconda.com/download/) (very large, includes lots
   of conda packages by default) or
   [miniconda](https://conda.io/miniconda.html) (much smaller, with only
   essential packages by default, but any conda package can be installed).
   
   
2. To update the materials:
   
    $ cd python-tutorial
    
    $ git pull
   
   
3. Create a new conda environment for this tutorial:

    *Installing from `yaml`:* **(recommended)**
    
        $ conda env create -f binder/environment.yml
    
    *Manually installing:*

        $ conda create -n tutorial --override-channels --strict-channel-priority -c conda-forge -c anaconda --yes jupyterlab=1
    
    *Activate the conda environment:*

        $ conda activate tutorial
    
    
4. Starting JupyterLab

    *Enter the following command in a new terminal window to start JupyterLab.*

        $ jupyter lab


5. Removing environment

    *You can delete the environment by using the following in a terminal prompt.*

        $ conda env remove --name tutorial --yes


## Useful links:

### Installation:
- [anaconda (python + libraries)](https://www.anaconda.com/distribution/)
- [Conda](https://conda.io/projects/conda/en/latest/index.html)
- [git](https://git-scm.com/)
- [python (only)](https://www.python.org/downloads/)

### Jupyter related:
- [Latex + jupyter](https://github.com/jupyterlab/jupyterlab-latex)
- [Git + jupyter](https://github.com/jupyterlab/jupyterlab-git)
- [ipywidgets + jupyter](https://github.com/jupyter-widgets/ipywidgets)
- [Awesome jupyter notebooks](https://github.com/markusschanta/awesome-jupyter)
- [Dask + jupyter](https://github.com/dask/dask-labextension)

### Reference guides
- [Numpy for matlab users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html)
- [Lecture series on scientific python by jrjohansson](https://github.com/jrjohansson/scientific-python-lectures)
- [Python datascience handbook online series](https://jakevdp.github.io/PythonDataScienceHandbook/index.html)
- [Scipy 2019 conference videos](https://www.youtube.com/user/EnthoughtMedia/videos)
- [Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

### Libraries + tutorials
- [Pangeo](https://pangeo.io/)
- [Dask](https://docs.dask.org/en/latest/)
- [Docker](https://hub.docker.com/)
- [Dask-jobqueue at PBSCluster](https://andersonbanihirwe.dev/talks/dask-jupyter-scipy-2019.html)

### Reproducability
- [Data Life-Cycle Management](https://www.dlcm.ch/)
- [OpenBIS docker container](https://hub.docker.com/r/openbis/debian-openbis)

