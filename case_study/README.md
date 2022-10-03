# gene_v.py
This file describes how to generate augmentation $\textbf{V}$ used in case study in section 3, including 5 adding ratios {0, 0.2, 0.4, 0.6, 0.8}.
# gene_heat_dis.py
In experimental analysis in section 4, we further contrast 9 existing augmentations with adjcency matrix. For {PPR Martix, PageRank, Eigenvector, Degree, Node Dropping, Subgraph, Edge Perturbation}, we utilize the original codes to generate. You can just generate them, put them into "../dataset/cora/" and rename as "cora_XXX.npz" (XXX is the chosen augmentation). \
In this file, we provide the generations of Heat Matrix and Distance. For the former, we correct the error in the provided code in MVGRL. For the later, we recode it by the instruction given in the MVGRL paper. More details can be found in this file.
# case_model.py
This file is our designed model for case study. You can use the following commands:
```
# test V
python case_model.py --dataset=cora --mode=0 --interval=low(high) --ratio=0(0.2, 0.4, 0.6, 0.8)

# test existing views
python case_model.py --dataset=cora --mode=-1 --view=dis/heat/diff/node/edge/subgraph/pr/evc/degree

# test A vs A
python case_model.py --dataset=cora --mode=1

# test A vs A^2
python case_model.py --dataset=cora --mode=12

# test A^2 vs A^2
python case_model.py --dataset=cora --mode=2
```
