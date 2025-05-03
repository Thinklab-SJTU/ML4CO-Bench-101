# DIMES

* Belonging to ``Heat4CO``.
* Type: ``GP-GCN-MAML``
    * Paradigm: Global Prediction (GP)
    * Model: Graph Convolution Network (GCN)
    * Training: Model-Agnostic Meta-(Reinforcement) Learning (MAML)
* Supported Problems
    * Traveling Salesman Problem (TSP)

## Setup
The following commands provide a list of critical dependency packages required for guaranteed reproducibility. We recommend that you maintain a strict version consistency.
```
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric==2.0.4
```
Next, you can locally install ``torch_sampling`` and compile ``dense_greedy`` as follows:
```
cd dimes/model
pip install ./torch_sampling
cd model/decoder
gcc ./dimes_greedy_dense.c -o dense_greedy.so -fPIC -shared
```
You may refer to [this page](https://github.com/DIMESTeam/DIMES/issues/4#issuecomment-1863087703) for details of some frequently raised issues about the installation of torch_sampling.

## Training
You can specify the detailed training settings in the following `.py` file and run:
```
python train_scripts/train_dimes_tsp.py
```
## Testing
You can specify the detailed testing settings in the following `.py` file and run:
```
python test_scripts/dimes/dimes_tsp.py
```

## Notice
If you see the following warning when running our code, just ignore it.

```
[W OperatorEntry.cpp:111] Warning: Registering a kernel (registered by RegisterOperators) for operator torch_scatter::segment_sum_csr for dispatch key (catch all) that overwrote a previously registered kernel with the same dispatch key for the same operator. (function registerKernel)
```

## Citation
```
@inproceedings{qiu2022dimes,
  title={{DIMES}: A differentiable meta solver for combinatorial optimization problems},
  author={Ruizhong Qiu and Zhiqing Sun and Yiming Yang},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```