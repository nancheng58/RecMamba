# 🚀Official PyTorch implementation of Rec-Mamba



- [🚀Official PyTorch implementation of Rec-Mamba](#official-pytorch-implementation-of-rec-mamba)
  - [✅abstract](#abstract)
  - [✅Performance](#performance)
  - [✅Efficiency](#efficiency)
  - [✅Requirements](#requirements)
  - [✅Usage](#usage)
  - [✅Citation](#citation)

## ✅abstract

## ✅Performance
**Performance in album:**  
![album](/element/album.png "album") 


**Performance in tracks:**  
![tracks](/element/tracks.png "tracks")


**Performance in artist:**  
![artist](/element/artist.png "artist") 


**Performance in KuaiRand:**  
![KuaiRand](/element/KuaiRand.png "KuaiRand") 



## ✅Efficiency
![Gpu and time](/element/Eff.png "Eff") 

## ✅Requirements

mamba_ssm   --> `1.1.1`  
matplotlib  --> `3.8.3` or newer  
numpy       --> `1.24.1` or newer  
torch       --> `2.1.1+cu118` or newer   
tqdm        --> `4.66.1` or newer  
wandb -->`0.12.18`  or newer  
GPU memory > `40G`

## ✅Usage

   `main.py`: This file provides a straightforward method for training, evaluating, and saving/loading Rec-Mamba models.  
   `model.py`: Python file containing implementations of `SasRec`, `Linrec`, and `Rec-Mamba`.  
   `utils.py`: Python file housing functions such as `data_partition` and `evaluation`.  
   `data_precess.py`: Python file used for preprocessing the `Kuairand` and `tracks` datasets.  
**data_process**
You can download the KuaiRand dataset from https://kuairand.com and LFM-1b_tracks dataset from https://github.com/RUCAIBox/RecSysDatasets    

1️⃣dataprecess  
`data_process.py` should be saved in the `data` folder   
You can run it with `python3 data_process.py`  
It takes about `30` minutes to process Kuairand, and about `60` minutes to process tracks.
![datasets](/element/dataset.png "Magic Gardens")   
After it, you can get different lenth's sequences saved in folder `data`   
for example:
```
1 2
1 3
1 5
1 6
It represents that User 1 interacted with items 2, 3, 5, and 6.
5 for valid and 6 for test

4 5
4 9
4 1000
4 8327
4 5
It represents that User 4 interacted with items 5, 9, 1000, 8327 and 5.
8327 for valid and 5 for test
```
2️⃣run for training, valid and test
  ```
  CUDA_VISIBLE_DEVICES=0 python3 -u main.py \ 
--dataset=KuaiRand5000 \ -----the file name you saved in folder data
--train_dir=5ksas2 \ ---- the dir where args,logs and .pth saved
--maxlen=5000 \   ---- the max sequence lenth
--dropout_rate=0.2 \  
--batch_size=32 \  
--lr=0.001 \      ---- learing rate, we used 0.0004 for KuaiRand,0.0002 for tracks
--neg_samples=1000 \  ---- test neg_sample number
--device=cuda \   --- device needed GPU memory > 40G for bact_size 256 sequence = 2k
--backbone=sas \  ---- you can choose sas,mamba,linrec
--name=5ksas2 \   ---- wandb project name
--hidden_units=50 > ./results/5ksas2.out ---- hidden state dimension
```


## ✅Citation












