# BS-RSC Dataset and AdaRSC Model

[CVPR2022] Learning Adaptive Warping for Real-World Rolling Shutter Correction

[Mingdeng Cao](https://github.com/ljzycmd), [Zhihang Zhong](https://zzh-tech.github.io/), [Jiahao Wang](https://scholar.google.com/citations?user=QjVR3UUAAAAJ), [Yinqiang Zheng](https://scholar.google.com/citations?user=JD-5DKcAAAAJ), [Yujiu Yang](https://scholar.google.com/citations?user=4gH3sxsAAAAJ)

---

[paper](https://arxiv.org/abs/2204.13886v1) **|** [dataset](https://drive.google.com/file/d/1h7UP1kci8zbg3TQp37J-imrvzlh2X6zn/view?usp=share_link) **|** [checkpoints](https://github.com/ljzycmd/BSRSC/releases/tag/v1.0.0) **|** [visual results](https://github.com/ljzycmd/BSRSC/releases/tag/v1.0.0)

> This paper proposes the first real-world rolling shutter (RS) correction dataset, BS-RSC, and a corresponding model to correct the RS frames in a distorted video. Mobile devices in the consumer market with CMOS-based sensors for video capture often result in rolling shutter effects when relative movements occur during the video acquisition process, calling for RS effect removal techniques. However, current state-of-the-art RS correction methods often fail to remove RS effects in real scenarios since the motions are various and hard to model. To address this issue, we propose a real-world RS correction dataset BS-RSC. Real distorted videos with corresponding ground truth are recorded simultaneously via a well-designed beam-splitter-based acquisition system. BS-RSC contains various motions of both camera and objects in dynamic scenes. Further, an RS correction model with adaptive warping is proposed. Our model can warp the learned RS features into global shutter counterparts adaptively with predicted multiple displacement fields. These warped features are aggregated and then reconstructed into high-quality global shutter frames in a coarse-to-fine strategy. Experimental results demonstrate the effectiveness of the proposed method, and our dataset can improve the model's ability to remove the RS effects in the real world.

## BS-RSC Dataset

We contribute the first real-world RSC dataset BS-RSC with various motions collected by a well-designed beam-splitter acquisition system, bridging the gap for real-world RSC tasks.

[[Google Drive](https://drive.google.com/file/d/1h7UP1kci8zbg3TQp37J-imrvzlh2X6zn/view?usp=share_link)]

## Adaptive Warping

We propose an adaptive warping module to exploit **multiple motion fileds strategy** and **adaptive warping based on self-attention mechanism** for high-quality GS frame restoration, mitigating inaccurate RS motion estimation and warping problems in existing CNN-based RSC methods.

## Quick Start

### Recommended Prerequisite

* Python 3.8
* PyTorch >= 1.5

We adopt the open-sourced deblurring framework [SimDeblur](https://github.com/ljzycmd/SimDeblur) for training and testing.

```bash
git clone https://github.com/ljzycmd/SimDeblur.git

# install SimDeblur package, problems may occur due the the CUDA version, please check
cd SimDeblur
bash Install.sh
```

### Testing

1. Downlaod the codes

```bash
git clone https://github.com/ljzycmd/BSRSC.git

cd BSRSC

# install cuda extension of Correlation layer
bash Install.sh
```

2. Prepare the datasets

* [BSRSC](https://drive.google.com/file/d/1h7UP1kci8zbg3TQp37J-imrvzlh2X6zn/view?usp=share_link)

* [FastecRS](https://github.com/ethliup/DeepUnrollNet)

3. Download checkpoints from [here](https://github.com/ljzycmd/BSRSC/releases/tag/v1.0.0) and run

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/adarsc_bsrsc.yaml ./PATH_TO_CKPT 
```

the outputs will be stored at ``workdir`` indicated at the config file.

### Training

1. Train the model wth BSRSC or FastecRS

```bash
# example of 4 GPUs training command
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh  ./PATH_TO_CONFIG  4
```

2. Train the model with customized data

You can format your data pairs into the structure of BSRSC dataset (the Dataset class implementation is available at `dataset/bsrsc.py`), and change the `root_gt` option in the config file (refer to the provided config file `configs/rscnet_bsrsc.yaml`) to the directory of your data.

## Visual Results

More results can be found at [here](https://github.com/ljzycmd/BSRSC/releases/tag/v1.0.0).

<details>
    <summary>Visual comparison on BS-RSC</summary>
    <img src="./docs/results_on_bsrsc.png">
</details>

<details>
    <summary>Visual comparison on synthetic dataset Fastec-RS</summary>
    <img src="./docs/results_on_fastecrs.png">
</details>


## Benchmarking

coming soon

## Citation

If the proposed model and dataset are useful for your research, please consider citing our paper

```bibtex
@InProceedings{Cao_2022_CVPR,
    author    = {Cao, Mingdeng and Zhong, Zhihang and Wang, Jiahao and Zheng, Yinqiang and Yang, Yujiu},
    title     = {Learning Adaptive Warping for Real-World Rolling Shutter Correction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17785-17793}
}
```

## Contact

If you have any questions about our project, please feel free to contact me at `mingdengcao [AT] gmail.com`.
