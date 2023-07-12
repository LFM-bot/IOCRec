# IOCRec
Pytorch implementation for paper: Multi-Intention Oriented Contrastive Learning for Sequential Recommendation (WSDM23).

We implement IOCRec in Pytorch and obtain quite similar results on Toys under the same experimental setting. The default hyper-parameters are set as the optimal values for Toys reported in the paper. Besides, the training log is available for reproduction.

```
2023-07-07 15:48:05 INFO     ------------------------------------------------Best Evaluation------------------------------------------------
2023-07-07 15:48:05 INFO     Best Result at Epoch: 33	 Early Stop at Patience: 10
2023-07-07 15:48:05 INFO     hit@5:0.4513	hit@10:0.5453	hit@20:0.6621	hit@50:0.7935	ndcg@5:0.3588	ndcg@10:0.3891	ndcg@20:0.4186	ndcg@50:0.4455	
2023-07-07 15:48:07 INFO     -----------------------------------------------------Test Results------------------------------------------------------
2023-07-07 15:48:07 INFO     hit@5:0.4022	hit@10:0.5005	hit@20:0.6205	hit@50:0.7594	ndcg@5:0.3145	ndcg@10:0.3462	ndcg@20:0.3765	ndcg@50:0.4048	
```
## Datasets
We provide Toys dataset.

## Quick Start
You can run the model with the following code:
```
python runIOCRec.py --dataset toys --eval_mode uni100 --embed_size 64 --k_intention 4 
```



