# Hypergraph Convolution Networks for Drug-Drug Interaction Prediction on the OGB-DDI Dataset
Winter 2021 CS224W Final Proj
Jay Liu (jliu99) & Sunwoo Kang (swkang73)

References:
* obgl-ddi dataset: https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi
* MGCN repo: https://github.com/sheldonresearch/MGCN
* SEAL_OGB: https://github.com/facebookresearch/SEAL_OGB


## Dependencies
Recommend using conda to set up environment with environment.yaml

## Baseline
SAGE: python baseline.py
GCN: python baseline_gcn.py

## Visualizing the Network
python vis_data.py 

## Visualizating Embeddings 
check our colab repository (contact swkang73 for access)

## Make 4-hop Neighborhood Hypergraph
python make_hypergraph.py

## Run HyperGCN
python run_hypergcn.py

## Run HyperSAGE
python run_hypersage.py
