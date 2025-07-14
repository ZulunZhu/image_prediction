# An Empirical Evaluation of Temporal Graph Benchmark on the remote sensing data


## Overview
This repository follows the [DyGLib_TGB](https://github.com/yule-BUAA/DyGLib_TGB) repository. 
We extend DyGLib_TGB to the remote sensing data.


## Dynamic Graph Learning Models
Eleven dynamic graph learning methods are included in DyGLib_TGB, including 
[JODIE](https://dl.acm.org/doi/10.1145/3292500.3330895), 
[DyRep](https://openreview.net/forum?id=HyePrhR5KX), 
[TGAT](https://openreview.net/forum?id=rJeW1yHYwH), 
[TGN](https://arxiv.org/abs/2006.10637), 
[CAWN](https://openreview.net/forum?id=KYPz4YsCPj), 
[EdgeBank](https://openreview.net/forum?id=1GVpwr2Tfdg), 
[TCL](https://arxiv.org/abs/2105.07944), 
[GraphMixer](https://openreview.net/forum?id=ayPPc0SyLv1), 
[DyGFormer](https://arxiv.org/abs/2303.13047),
[Persistent Forecast](https://arxiv.org/abs/2307.01026), and 
[Moving Average](https://arxiv.org/abs/2307.01026).


## Evaluation Tasks
Given the known coherence image, we regard these images as the edge features and the amplitude image as the node feature. The task is to predict the unknown edge feature.


## Environments
[PyTorch 2.0.1](https://pytorch.org/), 
[py-tgb](https://pypi.org/project/py-tgb/),
[numpy](https://pypi.org/project/numpy/), and
[tqdm](https://pypi.org/project/tqdm/).
You can use pip or conda tools to install these packages.

## Installation
Do the following in a clean conda environment (note that seaborn has been added as of 12 July 2025, have only tested its installation independently after pip install):  
```
$ source /home/tayw0049/miniconda3/etc/profile.d/conda.sh 
$ conda create -n image_prediction python=3.9 
$ conda activate image_prediction 
$ pip install py-tgb numpy==1.23.5 pandas==1.5.3 torch torchvision torchaudio matplotlib scikit-image 
$ conda install seaborn
```

Check installed packages: 
```
$ conda list 
packages in environment at /home/tayw0049/miniconda3/envs/image_prediction:                                                                                                                                           
Name                      Version                   Build  Channel          
_libgcc_mutex             0.1                        main
_openmp_mutex             5.1                       1_gnu
aiohappyeyeballs          2.6.1                    pypi_0    pypi
aiohttp                   3.11.13                  pypi_0    pypi
aiosignal                 1.3.2                    pypi_0    pypi 
args                      0.1.0                    pypi_0    pypi
async-timeout             5.0.1                    pypi_0    pypi
attrs                     25.1.0                   pypi_0    pypi
blas                      1.0                         mkl
bottleneck                1.4.2            py39ha9d4c09_0
brotli-python             1.0.9            py39h6a678d5_9
ca-certificates           2025.2.25            h06a4308_0
certifi                   2025.1.31                pypi_0    pypi
charset-normalizer        3.4.1                    pypi_0    pypi
clint                     0.5.1                    pypi_0    pypi
contourpy                 1.3.0                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
filelock                  3.17.0                   pypi_0    pypi
fonttools                 4.56.0                   pypi_0    pypi
freetype                  2.13.3               h4a9f257_0  
frozenlist                1.5.0                    pypi_0    pypi
fsspec                    2025.3.0                 pypi_0    pypi
idna                      3.10                     pypi_0    pypi
imageio                   2.37.0                   pypi_0    pypi
importlib-resources       6.5.2                    pypi_0    pypi
importlib_resources       6.4.0            py39h06a4308_0  
intel-openmp              2023.1.0         hdb19cb5_46306  
jinja2                    3.1.6                    pypi_0    pypi
joblib                    1.4.2                    pypi_0    pypi
jpeg                      9e                   h5eee18b_3  
kiwisolver                1.4.7                    pypi_0    pypi
lazy-loader               0.4                      pypi_0    pypi
lcms2                     2.16                 h92b89f2_1  
ld_impl_linux-64          2.40                 h12ee557_0  
lerc                      4.0.0                h6a678d5_0  
libdeflate                1.22                 h5eee18b_0  
libffi                    3.4.4                h6a678d5_1  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libpng                    1.6.39               h5eee18b_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.7.0                hde9077f_0  
libwebp-base              1.3.2                h5eee18b_1  
lz4-c                     1.9.4                h6a678d5_1  
markupsafe                3.0.2                    pypi_0    pypi
matplotlib                3.9.4                    pypi_0    pypi
matplotlib-base           3.9.2            py39hbfdbfaf_1  
mkl                       2023.1.0         h213fc3f_46344  
mkl-service               2.4.0            py39h5eee18b_2  
mkl_fft                   1.3.11           py39h5eee18b_0  
mkl_random                1.2.8            py39h1128e8f_0  
mpmath                    1.3.0                    pypi_0    pypi
multidict                 6.1.0                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
networkx                  3.2.1                    pypi_0    pypi
numexpr                   2.10.1           py39h3c60e43_0  
numpy                     1.23.5                   pypi_0    pypi
numpy-base                2.0.1            py39hb5e798b_1  
nvidia-cublas-cu12        12.4.5.8                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.4.127                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.4.127                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.4.127                 pypi_0    pypi
nvidia-cudnn-cu12         9.1.0.70                 pypi_0    pypi
nvidia-cufft-cu12         11.2.1.3                 pypi_0    pypi
nvidia-curand-cu12        10.3.5.147               pypi_0    pypi
nvidia-cusolver-cu12      11.6.1.9                 pypi_0    pypi
nvidia-cusparse-cu12      12.3.1.170               pypi_0    pypi
nvidia-cusparselt-cu12    0.6.2                    pypi_0    pypi
nvidia-nccl-cu12          2.21.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.4.127                 pypi_0    pypi
nvidia-nvtx-cu12          12.4.127                 pypi_0    pypi
openjpeg                  2.5.2                h0d4d230_1  
openssl                   3.0.16               h5eee18b_0  
packaging                 24.2             py39h06a4308_0  
pandas                    1.5.3                    pypi_0    pypi
pillow                    11.1.0           py39hac6e08b_1  
pip                       25.0             py39h06a4308_0  
propcache                 0.3.0                    pypi_0    pypi
psutil                    7.0.0                    pypi_0    pypi
py-tgb                    2.0.0                    pypi_0    pypi
pyparsing                 3.2.1                    pypi_0    pypi
python                    3.9.21               he870216_1  
python-dateutil           2.9.0post0       py39h06a4308_2  
python-tzdata             2025.2             pyhd3eb1b0_0  
pytz                      2025.1                   pypi_0    pypi
readline                  8.2                  h5eee18b_0  
requests                  2.32.3                   pypi_0    pypi
scikit-image              0.24.0                   pypi_0    pypi
scikit-learn              1.6.1                    pypi_0    pypi
scipy                     1.13.1                   pypi_0    pypi
seaborn                   0.13.2           py39h06a4308_3  
setuptools                75.8.0           py39h06a4308_0  
six                       1.17.0           py39h06a4308_0  
sqlite                    3.45.3               h5eee18b_0  
sympy                     1.13.1                   pypi_0    pypi
tbb                       2021.8.0             hdb19cb5_0  
threadpoolctl             3.5.0                    pypi_0    pypi
tifffile                  2024.8.30                pypi_0    pypi
tk                        8.6.14               h39e8969_0  
torch                     2.6.0                    pypi_0    pypi
torch-geometric           2.6.1                    pypi_0    pypi
torchaudio                2.6.0                    pypi_0    pypi
torchvision               0.21.0                   pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
triton                    3.2.0                    pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
tzdata                    2025a                h04d1e81_0  
unicodedata2              15.1.0           py39h5eee18b_1  
urllib3                   2.3.0                    pypi_0    pypi
wheel                     0.45.1           py39h06a4308_0  
xz                        5.6.4                h5eee18b_1  
yarl                      1.18.3                   pypi_0    pypi
zipp                      3.21.0           py39h06a4308_0  
zlib                      1.2.13               h5eee18b_1  
zstd                      1.5.6                hc292b87_0
```

As part of the tgb installation, other packages such as pytorch, tqdm, scikit-learn will be installed along with tgb too. However, other packages such as torch, matplotlib, and scikit-image will not be installed and need to be specified during installation. 

Note that tgb can only be installed using pip and requires pandas <2.00 >=1.5.3. Note that pandas 1.5.3 is only compatible with earlier versions of numpy, e.g. numpy 1.23.5. By default, tgb would install a later numpy version 2.0.2, so the numpy and pandas versions need to be specified when installing tgb. See more about tgb here: https://github.com/shenyangHuang/TGB. If there is a numpy and pandas conflict, there would be errors when running a python code which imports scikit-learn, pandas or tgb


## Executing Scripts
You can change the "--model_name" parameter into other models


### Scripts for Dynamic Link Property Prediction
python image_link_prediction.py --dataset_name tgbl-wiki --model_name TCL --patch_size 1 --max_input_sequence_length 32 --num_runs 5 --gpu 0
