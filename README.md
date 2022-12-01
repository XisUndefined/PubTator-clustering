# How to Run the clustering

## Clone Repository
The section below describes how to clone the repository to perform the clustering algorithm. To install the dependencies for clustering, you need to clone the repository by running the command as follows:
```sh
$ git clone https://github.com/XisUndefined/PubTator-clustering.git
$ cd PubTator-clustering
```

## Set-up Environment

### Conda

1. Install the [conda](https://conda.io) environment.
2. Create the python environment by running:
```sh
conda create --name env-name python==3.7.12
```
3. Activate with `conda activate env-name`
4. Install packages via pip by running the following:
```sh
pip install -r requirements.txt
```

### Pip

1. Make sure you have python version **3.7.12** installed.
2. Install packages by running the following:
```sh
pip install -r requirements.txt
```

## Execution
To start the clustering process run the following command:
```sh
python execute.py
```