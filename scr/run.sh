#!/bin/bash
python main.py   --dataset cora --nlayers=2 --lr_feat=1e-3 --lr_adj=1e-3 --r=0.058  --sgc=0 --dis=mse --gpu_id=0 --one_step=1  --epochs=2000  --attack=random --ptb_rate=1.0 --knn=2
