# A Generalized Loss Function for Crowd Counting and Localization

## Data preparation
The dataset can be constructed followed by [Bayesian Loss](https://github.com/ZhihengCV/Bayesian-Crowd-Counting).

## Pretrained model
The pretrained model can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1TJF2IeFPoeLzqNXKXXXK8nPH62HijZaS?usp=sharing).

## Test

```
python test.py --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT
```

## Train

```
python train.py --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT
```

### Citation
If you use our code or models in your research, please cite with:

```
@InProceedings{Wan_2021_CVPR,
    author    = {Wan, Jia and Liu, Ziquan and Chan, Antoni B.},
    title     = {A Generalized Loss Function for Crowd Counting and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    pages     = {1974-1983}
}
```

### Acknowledgement
We use [GeomLoss](https://www.kernel-operations.io/geomloss/) package to compute transport matrix. Thanks for the authors for providing this fantastic tool. The code is slightly modified to adapt to our framework.
