# 🚀Official PyTorch implementation of Rec-Mamba



- [🚀Official PyTorch implementation of Rec-Mamba](#official-pytorch-implementation-of-rec-mamba)
  - [✅abstract](#abstract)
  - [✅Performance](#performance)
  - [✅Efficiency](#efficiency)
  - [✅Requirements](#requirements)
  - [✅Usage](#usage)
  - [✅Citation](#citation)

## ✅Abstract
Sequential Recommenders have been widely applied in various online services, aiming to model users' dynamic interests from their sequential interactions. With users increasingly engaging with online platforms, vast amounts of lifelong user behavioral sequences have been generated. However, existing sequential recommender models often struggle to handle such lifelong sequences. The primary challenges stem from computational complexity and the ability to capture long-range dependencies within the sequence. Recently, a state space model featuring a selective mechanism (i.e., Mamba) has emerged. In this work, we investigate the performance of Mamba for lifelong sequential recommendation (i.e., length>=2k). More specifically, we leverage the Mamba block to model lifelong user sequences selectively. We conduct extensive experiments to evaluate the performance of representative sequential recommendation models in the setting of lifelong sequences. Experiments on two real-world datasets demonstrate the superiority of Mamba. We found that RecMamba achieves performance comparable to the representative model while significantly reducing training duration by approximately 70% and memory costs by 80%.

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
![Gpu and time](/element/effiency.png "Eff") 

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
If you find our codes and datasets useful for your research, please cite:
```
@misc{yang2024uncovering,
      title={Uncovering Selective State Space Model's Capabilities in Lifelong Sequential Recommendation}, 
      author={Jiyuan Yang and Yuanzi Li and Jingyu Zhao and Hanbing Wang and Muyang Ma and Jun Ma and Zhaochun Ren and Mengqi Zhang and Xin Xin and Zhumin Chen and Pengjie Ren},
      year={2024},
      eprint={2403.16371},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```












