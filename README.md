# IOCRec
Pytorch implementation for paper: Multi-Intention Oriented Contrastive Learning for Sequential Recommendation (WSDM23).

We implement IOCRec in Pytorch and obtain quite similar results on Toys under the same experimental setting. The default hyper-parameters are set as the optimal values for Toys reported in the paper. Besides, the training log is available for reproduction.

## Datasets
We provide Toys dataset.

## Quick Start
You can run the model with the following code:
```
python runIOCRec.py --dataset toys --embed_size 64 --k_intention 4 
```



