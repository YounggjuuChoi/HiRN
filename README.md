# HiRN: Hierarchical Recurrent Neural Network for Video Super-Resolution (VSR) using Two-Stage Feature Evolution
#### Young-Ju Choi and Byung-Gyu Kim
#### Intelligent Vision Processing Lab. (IVPL), Sookmyung Women's University, Seoul, Republic of Korea
----------------------------
#### This repository is the official PyTorch implementation of the paper published in _Applied Soft Computing (Elsevier)_.
[![paper](https://img.shields.io/badge/paper-PDF-<COLOR>.svg)](https://www.sciencedirect.com/science/article/pii/S1568494623004404)

----------------------------
## Summary of paper
#### Abstract
> _The aim of video super-resolution (VSR) is generate the high-resolution (HR) frames from their lowresolution
(LR) counterparts. As one of the fundamental module of VSR, propagation process provides
the path of feature map and specifies how the feature map is leveraged. In the recurrent propagation,
the latent features can be propagated and aggregated. Therefore, adopting the recurrent strategy can
resolve the limitation of sliding-window-based local propagation. Recently, bi-directional recurrent
propagation-based latest methods have achieved powerful performance in VSR. However, existing
bi-directional frameworks have structured by combining forward and backward branches. These
structures cannot propagate and aggregate previous and future latent features of current branch. In this
study, we suggest the hierarchical recurrent neural network (HiRN) based on feature evolution. The
proposed HiRN is designed based on the hierarchical recurrent propagation and residual block-based
backbone with temporal wavelet attention (TWA) module. The hierarchical recurrent propagation
consists of two stages to combine advantages of low frame rate-based forward and backward schemes,
and multi-frame rate-based bi-directional access structure. The proposed methods are compared with
state-of-the-art (SOTA) methods on the benchmark datasets. Experiments show that the proposed
scheme achieves superior performance compared with SOTA methods. In particular, the proposed
HiRN achieves better performance than all compared methods in terms of SSIM on Vid4 benchmark.
In addition, the proposed HiRN surpasses the existing GBR-WNN by a significant 3.03 dB in PSNR on
REDS4 benchmark with fewer parameters._
>

#### Network Architecture
<p align="center">
  <img width="900" src="./images/img1.PNG">
</p>

#### Experimental Results
<p align="center">
  <img width="900" src="./images/img2.PNG">
</p>

<p align="center">
  <img width="900" src="./images/img3.PNG">
</p>

<p align="center">
  <img width="900" src="./images/img4.PNG">
</p>

----------------------------
## Getting Started
#### Dependencies and Installation
- Anaconda3
- Python == 3.6
    ```bash
    conda create --name gbrwnn python=3.6
    ```
- [PyTorch](https://pytorch.org/) (NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads))
    
    Trained on PyTorch 1.8.1 CUDA 10.2
    ```bash
    conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
    ```
    
- tqdm, pyyaml, tensorboard, opencv-python, lmdb
    ```bash
    conda install -c conda-forge tqdm pyyaml tensorboard
    pip install opencv-python
    pip install lmdb
    ```


#### Dataset Preparation
We used [Vimeo90K](https://arxiv.org/pdf/1711.09078.pdf) dataset for training and [Vid4](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6549107), [REDS4](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Deblurring_and_Super-Resolution_Dataset_and_CVPRW_2019_paper.pdf), [SPMCS](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tao_Detail-Revealing_Deep_Video_ICCV_2017_paper.pdf), [DAVIS-2019](https://arxiv.org/pdf/1905.00737.pdf) datasets for testing.

- Prepare for Vimeo90K

    1) Please refer to **[Dataset.md](https://github.com/YounggjuuChoi/Deep-Video-Super-Resolution/blob/master/Doc/Dataset.md)** in our **[Deep-Video-Super-Resolution](https://github.com/YounggjuuChoi/Deep-Video-Super-Resolution)** repository for more details.

    2) Download dataset from the [official website](http://toflow.csail.mit.edu/).
    
    3) Put the dataset in ./datasets/
 
    4) Generate LR data
       
       Run in ./codes/data_processing_scripts/
       ```bash
       python generate_LR_Vimeo90K.py
       ```
       
    5) Generate LMDB
 
       Run in ./codes/data_processing_scripts/
       ```bash
       python generate_lmdb_Vimeo90K.py
       ```
    
- Prepare for Vid4

    1) Please refer to **[Dataset.md](https://github.com/YounggjuuChoi/Deep-Video-Super-Resolution/blob/master/Doc/Dataset.md)** in our **[Deep-Video-Super-Resolution](https://github.com/YounggjuuChoi/Deep-Video-Super-Resolution)** repository for more details.
 
    2) Download dataset from [here](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA).
    
    3) Put the dataset in ./datasets/
 
    4) Generate LR data

       Run in ./codes/data_processing_scripts/ 

       ```bash
       python generate_LR_Vid4.py
       ```

- Prepare for REDS4

    1) Please refer to **[Dataset.md](https://github.com/YounggjuuChoi/Deep-Video-Super-Resolution/blob/master/Doc/Dataset.md)** in our **[Deep-Video-Super-Resolution](https://github.com/YounggjuuChoi/Deep-Video-Super-Resolution)** repository for more details.
 
    2) Download dataset from the [official website](https://seungjunnah.github.io/Datasets/reds.html).
    
    3) Put the dataset in ./datasets/

- Prepare for SPMCS

    1) Download dataset from [here](https://github.com/jiangsutx/SPMC_VideoSR).
    
    2) Put the dataset in ./datasets/
 
    3) Generate LR data

       Run in ./codes/data_processing_scripts/ 

       ```bash
       python generate_LR_SPMCS.py
       ```

- Prepare for DAVIS-2019

    1) Download dataset from the [official website](https://davischallenge.org/challenge2019/unsupervised.html).
    
    2) Put the dataset in ./datasets/
 
    3) Generate LR data

       Run in ./codes/data_processing_scripts/ 

       ```bash
       python generate_LR_DAVIS.py
       ```


#### Model Zoo
Pre-trained models are available in below link.

[![google-drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1L379h1XRjGC2Bwh5At9MUTbRYm0OXHiN?usp=sharing)


----------------------------
## Training
Run in ./codes/
- GBR-WNN-L

    Using single GPU
    ```bash
    python train.py -opt options/train/train_GBRWNN_L.yml
    ```
    Using multiple GPUs (nproc_per_node means the number of GPUs)
    with setting CUDA_VISIBLE_DEVICES in .yml file
    
    For example, set 'gpu_ids: [0,1,2,3,4,5,6,7]' in .yml file for 8 GPUs 
    
    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/train_GBRWNN_L.yml --launcher pytorch
    ```
    
- GBR-WNN-M

    Using single GPU
    ```bash
    python train.py -opt options/train/train_GBRWNN_M.yml
    ```
    Using multiple GPUs (nproc_per_node means the number of GPUs)
    with setting CUDA_VISIBLE_DEVICES in .yml file
    
    For example, set 'gpu_ids: [0,1,2,3,4,5,6,7]' in .yml file for 8 GPUs 
    
    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/train_GBRWNN_M.yml --launcher pytorch
    ```

- GBR-WNN-S

    Using single GPU
    ```bash
    python train.py -opt options/train/train_GBRWNN_S.yml
    ```
    Using multiple GPUs (nproc_per_node means the number of GPUs)
    with setting CUDA_VISIBLE_DEVICES in .yml file
    
    For example, set 'gpu_ids: [0,1,2,3,4,5,6,7]' in .yml file for 8 GPUs 
    
    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/train_GBRWNN_S.yml --launcher pytorch
    ```

----------------------------
## Testing
Run in ./codes/

```bash
python test.py
```

You can test the GBR-WNN-L, GBR-WNN-M, GBR-WNN-S models under Vid4, REDS4, SPMCS, DAVIS-2019 test datasets by modifying the _'model_mode'_ and _'data_mode'_ in source code.
    
----------------------------
## Citation
    @article{choi2022group,
      title={Group-based bi-directional recurrent wavelet neural network for efficient video super-resolution (VSR)},
      author={Choi, Young-Ju and Lee, Young-Woon and Kim, Byung-Gyu},
      journal={Pattern Recognition Letters},
      volume={164},
      pages={246--253},
      year={2022},
      publisher={Elsevier}
    }
    
----------------------------
## Acknowledgement
The codes are heavily based on [EDVR](https://github.com/xinntao/EDVR) and [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome works.

```bash
EDVR : 
Wang, Xintao, et al. "Edvr: Video restoration with enhanced deformable convolutional networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2019.
```

```bash
BasicSR :
@misc{basicsr,
  author =       {Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C.K. Chan and Chen Change Loy and Chao Dong},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/XPixelGroup/BasicSR}},
  year =         {2022}
}
```
