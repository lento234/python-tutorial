# Python tutorials

A compilation of useful jupyter notebook files for learning python.

**Online Notebooks:**

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lento234/python-tutorial/master?urlpath=lab)
[![Azure](https://notebooks.azure.com/launch.svg)](https://notebooks.azure.com/import/gh/lento234/python-tutorial)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lento234/python-tutorial/blob/master/04_numerical_python/04_numpy.ipynb)

**Continuous integration:**

[![Build Status](https://travis-ci.com/lento234/python-tutorial.svg?branch=master)](https://travis-ci.com/lento234/python-tutorial)

**Slides**
1. [Introduction (15th July, 2019)](https://bit.ly/2Y9zc6v)
2. Basic python (TBD)


**References (derived and modified from various sources):**

- [Scipy 2019 jupyterlab tutorial](https://github.com/jupyterlab/scipy2019-jupyterlab-tutorial)
- [Lecture series on scientific python by jrjohansson](https://github.com/jrjohansson/scientific-python-lectures)
- [Scipy Lecture notes](https://scipy-lectures.org/)
- [Quantitative Big Imaging 2019 by Kevin Mader (ETHZ)](https://github.com/kmader/Quantitative-Big-Imaging-2019)
- [Dask examples tutorial](https://github.com/dask/dask-examples)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)


## Software installation

1. Install python (**python 3**) on your computer.

    From [miniconda](https://conda.io/miniconda.html) containing only the essentials such as the `conda` package  manager and `python`. (**Recommended**) 
    
    *or*

    From [anaconda distribution](https://www.anaconda.com/download/) containing all the necessary libaries for scientific research. Warning: very large, includes lots of conda packages by default.
   
2. Download this python tutorial github repository on to your computer.

    *Clone this `github` repository locally to you computer:*
      
        $ git clone https://github.com/lento234/python-tutorial.git
      
    *Change your directory to the tutorial folder:*
      
        $ cd python-tutorial
   
3. Create a new conda environment for this tutorial.

    *Installing from the `yaml` file included in this repository:*
    
        $ conda env create -f binder/environment.yml
    
    *Activate the conda `tutorial` environment:*

        $ conda activate tutorial
    
4. Starting JupyterLab locally.

    *Enter the following command in a new terminal window to start JupyterLab:*

        $ jupyter lab

5. Deactivate your environment when you are done.

        $ conda deactivate 

6. (*optional*) Removing your `tutorial` environment if you don't want it anymore.

    *You can delete the environment by using the following in a terminal prompt:*

        $ conda env remove --name tutorial --yes


## Useful links:

### Installation:
- [git](https://git-scm.com/)
- [Conda](https://conda.io/projects/conda/en/latest/index.html)
- [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [anaconda (python + libraries)](https://www.anaconda.com/distribution/)
- [python (only)](https://www.python.org/downloads/)
- [Visual studio code](https://code.visualstudio.com/)

### Reference guides:
- [Numpy for matlab users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html)
- [Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
- [Lecture series on scientific python by jrjohansson](https://github.com/jrjohansson/scientific-python-lectures)
- [Python datascience handbook online series](https://jakevdp.github.io/PythonDataScienceHandbook/index.html)
- [Scipy 2019 conference videos](https://www.youtube.com/user/EnthoughtMedia/videos)

### Jupyter related:
- [Latex + jupyter](https://github.com/jupyterlab/jupyterlab-latex)
- [Git + jupyter](https://github.com/jupyterlab/jupyterlab-git)
- [ipywidgets + jupyter](https://github.com/jupyter-widgets/ipywidgets)
- [Awesome jupyter notebooks](https://github.com/markusschanta/awesome-jupyter)
- [Dask + jupyter](https://github.com/dask/dask-labextension)

### Libraries + tutorials:
- [Pangeo](https://pangeo.io/)
- [Dask](https://docs.dask.org/en/latest/)
- [Docker](https://hub.docker.com/)
- [Dask-jobqueue at PBSCluster](https://andersonbanihirwe.dev/talks/dask-jupyter-scipy-2019.html)
- [Awesome python](https://awesome-python.com)

### Reproducability in science:
- [Data Life-Cycle Management](https://www.dlcm.ch/)
- [OpenBIS docker container](https://hub.docker.com/r/openbis/debian-openbis)

