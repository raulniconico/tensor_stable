## Presentation
This package is a part of project "Apprentissage profond pour l’identification de molécules dans le milieu interstellaire."
at laboratory ISMO. Financed by foundation DATAIA 

<!-- toc -->
- [1. Installlation](#1-installation)
  - [1.1 pip](#11-pip) 
  - [1.2 Conda](#12-conda)
  - [1.3 Compiling](#13-local-compiling)
- [2. Package structure](#2-package-structure)
- [3. HTD & HT-FBR Usage API](#3-htd--ht-fbr-usage-api)
  - [3.1 HTD user guide](#31-htd-user-guide)
  - [3.2 HT-FBR user guide](#32-ht-fbr-user-guide)
- [4 Optimization of HTD & HT-FBR](#4-optimization-of-ht-fbr)
  - [4.1 Initialization of the database](#41-Initialization-of-the-database)
  - [4.2 Initialization of Optim Btree](#42-initialization-of-an-optim_btree)
  - [4.3 Training loop](#43-training-loop)
<!-- tocstop -->


## 1. Installation

```bash
git clone -b master https://<TOKEN>@github.com/raulniconico/tensor_stable.git
```

### 1.1 pip
cd to the root of the repository, run:

```bash
pip install -e .
```

### 1.2 Conda
You can create a conda environment, first activate the environment, then install the package at the root:  

```bash
conda create --name HTD python=3.10
conda activate HTFBR
pip install -e .
conda env export > environment_HTFBR.yml
```

### 1.3 Local compiling
```bash
chmod +x build.sh
. build.sh
```

## 2 Package structure
The tensor(tk) package contains three subprojects: 
- **nuwa**: codes for graph computation as node, binary tree etc.
- **HTD**: HTD and HT-FBR binary tree. Decomposition API.
- **optim**: optimization of HTD and HT-FBR. Including training toolkit.

The dependency of the package:

```
tensor(tk)/
├── nuwa/
├── HTD/
├── optim/
```

## 3. HTD & HT-FBR Usage API
We provide examples for decomposition or training a HTD/HT-FBR binary tree to fit a tensor

### 3.1 HTD user guide
The class of HTD binary tree is called HT_Btree

Create an empty binary tree

```python
from tensor.HTD import HT_Btree

shape = (10, 10, 10, 10)
HTD_Btree = HT_Btree.empty(shape)
```

A more usual using case is using function **hsvd()** to decompose a given tensor:

Function hsvd() takes two parameters: 

- A: np.ndarray, dense tensor to be decomposed
- k: tuple, node rank used in each level (not included the root node. It is always to be rank 1) 

The structure of the binary tree:

```
       0       level 0
     /   \
    1     2    level 1
   / \   / \
  3   4 5   6  level 2
```

To reconstruct from binary tree, package provides function reconstruction_from_tree()

```python
from tensor.HTD import reconstruction_from_tree

A_HTD_recon = reconstruction_from_tree(btree=HTD_btree)
```

Note that, the reconstructed tensor is explicitly **vectorized**. 
Use reshape() to get a tensor with same shape as the original tensor. 

Here's a full example:

```python
from tensor.HTD import hsvd
import numpy as np

# A is the tensor to be decomposied 
A = np.random.randn(10, 10, 10, 10)

# node rank. k[0]: node rank used in btree level 1, k[1] used in level 2 (leaf nodes)
k = (10, 10)  # (node rank in l1, node rank in l2)

# hsvd returns a HTD binary tree with rank k
btree_HTD = hsvd(A, k)
```

### 3.2 HT-FBR user guide

API for HT-FBR is the same.
HT-FBR uses polynomials as analytical basis functions. The domain of the finite basis is implicitly defined by
the points for interpolation. The number of these points is same as the points in each dimension of the tensor. 

```python
n = 10

# A is a random tensor with n points in each dimension
A = np.random.randn(n, n, n ,n)

# Points for interpolation. The number of each should be same as A.shape
# "mesh" implicitly defines the domain of the finite basis
mesh = (np.linspace(-50, 50, n),
        np.linspace(-50, 50, n),
        np.linspace(-50, 50, n),
        np.linspace(-50, 50, n))

k = (10, 10) # node rank

btree_HT_FBR = hsvd(A, k=k, poly="herm", deg=8, coor=mesh)
A_HT_FBR_recon = reconstruction_from_tree(btree=btree_HT_FBR, coor=mesh)
```


## 4. Optimization of HT-FBR
Hereafter is the optimization for HT-FBR. All the factors and weights are randomly generalized. The nuwa schema uses a
batched strategy for training. The input and label are splited into pieces of B. For training the binary tree, one need
tensor.optim.Optim_Btree . To be more precise, tensor.optim.HTD_Optim_Btree and tensor.optim.HT_FBR_Optim_Btree .

The opitimization of HT-FBR follows the following steps:
1. Initialization of the database
2. Initialization of an Optim Binary tree (Optim_Btree)
3. Forward propagate the batched training data into Optim Btree
4. Evaluation the quality of the prediction (loss function)
5. Backward propagate the gradient
6. Optimizer updates the weights (transfer tensors)
7. If the loss converges, exit. Else, return to 3.

### 4.1 Initialization of the database
The database is suggested to use tensor.optim.Dataset and tensor.optim.Dataloader. 
The input coordinates should have a shape of (N, D) where N is the total number of the samplings while D is the dimension of
the input coordinates. The label should be (N, 1) to align with the input. Note that shape (N, 1) is not same as (N, ) 
in numpy convention. Make sure the second dimension is 1 not default.

The Dataloader will split the training data by the batches. In each training batch it returns a batched input with shape 
(B, D) and label (B, 1).   

Here's an example for creating a Dataset and Dataloader

```python
from tensor.optim import Dataset, Dataloader
import numpy as np

# a training input with 10,000 samplings and 4 dimension
train_coor = np.random.randn(10000, 4)
# training energy has a shape (10000, 1)
train_energy = np.random.randn(10000, 1)

# set batch size
batch_size = 1000

# Initial Dataset and Dataloader
train_set = Dataset(train_coor, train_energy)
train_loader = Dataloader(train_set, batch_size=batch_size, shuffle=True)
```

### 4.2 Initialization of an Optim_Btree
Before the training creates a HT_FBR_Optim_Btree class. It takes two positional arguments: k and shape. They decide the 
actual shape of the factors and of the weights. All the args & kwargs show following:

1. k: Union[tuple, np.ndarray],
2. shape: Union[tuple, np.ndarray],
3. kernel: str = "concatenate". Kernel for propagation (concatenate)
4. poly: str = "none". The polynomial to be used 
5. deg: int = None. Degree of polynomial 
6. coor: tuple = None. Mesh for interpolation 
7. dtype=np.float64

```python
from tensor.optim import HT_FBR_Optim_Btree

# node rank
k = (10, 10)
# The expected shape 0 for the factor of each dimension
shape = (10, 10, 10, 10)

btree = HT_FBR_Optim_Btree(k=k, shape=shape)
# Multiple a factor of 10 to the weights. It is suggested when fitting PES
btree.init_factor = 10
# Set kernle of HT-FBR
btree.kernel = "concatenate"
# Use degree 7 Hermite polynomial as FBR 
btree.deg = 7
btree.poly = "herm"
# Give polynomial anchor mesh
btree.coor = mesh
# Initial HT-FBR binary tree
btree.init()
```

### 4.3 Training loop
During the training loop:

**Prediction:** predict = btree.forward(feature)

**Loss function:** loss = RMSE(predict, label)

**Get root grad:** RMSE_grad(predict, label)

**Update grad:** btree.root.update_cache({"grad_0": grad})

**Backpropagation:** btree.backward()

**Optimization** optimizer.step()



```python
from tqdm import tqdm

# initial training 
NUM_EPOCHS = 10000
train_losses = []
test_losses = []
tepoch = tqdm(range(1, NUM_EPOCHS + 1), unit="batch")

# training
for epoch in tepoch:
    
    tepoch.set_description(f"Epoch {epoch}")
    # In each epoch, shuffle the training set
    train_loader.init()
    loss = 0
    
    for i, batch in enumerate(train_loader):
        
        feature, label = batch 
        # 1. Input of the HT-FBR tree is coordinates
        predict = btree.forward(feature) 
        # 2. Compute RMSE loss
        loss += RMSE(predict, label)
        # 3. Set gradient of the loss function to the root of HT-FBR tree
        #    backward propagate the gradient 
        grad = RMSE_grad(predict, label)
        btree.root.update_cache({"grad_0": grad})
        btree.backward()
        # 4. optimize the weights
        optimizer.step()

    train_losses.append(loss/len(train_loader))
    tepoch.set_postfix(loss=loss/len(train_loader))

```


