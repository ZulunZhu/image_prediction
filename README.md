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


## Executing Scripts
You can change the "--model_name" parameter into other models
### Scripts for Dynamic Link Property Prediction
python image_link_prediction.py --dataset_name tgbl-wiki --model_name TCL --patch_size 1 --max_input_sequence_length 32 --num_runs 5 --gpu 0
