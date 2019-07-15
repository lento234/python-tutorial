
# 01 - Installing anaconda
-----

- **python distribution**
- **python package management**
- **Resolve dependencies** 

#### Specifications:
- Linux, mac, windows (?)
- Anaconda 2019.03

![figure](images/anaconda.png)

### **Step 1: Install anaconda**

https://docs.anaconda.com/anaconda/install/

### **Step 2: Verify installation**

- open python prompt: `ipython`

## **Setup conda environments**
-----

https://conda.io/projects/conda/en/latest/

![figure](images/conda.png)

**Macking a new environment**

*Example environment `test` installing package `numpy 1.14`*

```bash
conda create -n test numpy=1.14 # specific package
```

*List conda environments*

```bash
conda info --envs
```

*Initialize* (__must restart terminal__)


```bash
conda init
```

*Activate environment `test`*

```bash
conda activate test
```

## **Setup conda environments**
-----

https://conda.io/projects/conda/en/latest/


### **Step1 : Install jupyterlab**

```bash
conda create -n tutorial --override-channels --strict-channel-priority -c conda-forge -c anaconda --yes jupyterlab=1 ipywidgets nodejs pip cookiecutter pandas=0.24 matplotlib
```

### **Step 2: Start jupyterlab**

```bash
jupyter lab
```
