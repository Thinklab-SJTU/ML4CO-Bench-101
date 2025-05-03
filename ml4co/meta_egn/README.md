# Meta-EGN

* Belonging to ``Heat4CO``.
* Type: ``GP-GCN-MAML``
    * Paradigm: Global Prediction (GP)
    * Model: Graph Convolution Network (GCN)
    * Training: Model-Agnostic Meta-(Unsupervised) Learning (MAML)
* Supported Problems
    * Maximum Clique (MCl)
    * Maximum Cut (MCut)
    * Maximum Independent Set (MIS)
    * Minimum Vertex Cover (MVC)

## Setup
The following commands provide a list of critical dependency packages required for guaranteed reproducibility. We recommend that you maintain a strict version consistency.

```
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --no-index torch-sparse==0.6.11 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --no-index torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --no-index torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-geometric==1.7.2
```

## Training
You can specify the detailed training settings in the corresponding `.py` files and run:
```
python train_scripts/train_meta_egn_mcl.py
python train_scripts/train_meta_egn_mis.py
python train_scripts/train_meta_egn_mvc.py
python train_scripts/train_meta_egn_mcut.py
```

## Testing
You can specify the detailed testing settings in the corresponding `.py` files and run:
```
python test_scripts/meta_egn/meta_egn_mcl.py 
python test_scripts/meta_egn/meta_egn_mis.py 
python test_scripts/meta_egn/meta_egn_mvc.py 
python test_scripts/meta_egn/meta_egn_mcut.py 
```

## Citation
```
@article{wang2023unsupervised,
  title={Unsupervised Learning for Combinatorial Optimization Needs Meta-Learning},
  author={Wang, Haoyu and Li, Pan},
  journal={International Conference on Learning Representations},
  year={2023}
}
```