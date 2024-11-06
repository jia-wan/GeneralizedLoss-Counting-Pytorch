# A Generalized Loss Function for Crowd Counting and Localization

## Data preparation
The dataset can be constructed followed by [Bayesian Loss](https://github.com/ZhihengCV/Bayesian-Crowd-Counting).

## Pretrained model
The pretrained model can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1TJF2IeFPoeLzqNXKXXXK8nPH62HijZaS?usp=sharing).

## Traina and Test

```
1. generate segmentations for each image with SEEM as pseudo labels (run_gen_sam_labels.py)
2. train a counting network with pseudo labels (run_train_counter.py)
3. predict the location with the counting network (run_resam.py)
4. use the prediction as prompts to generate new masks with SEEM + point prompt (run_resam.py)
5. merge the new masks with the old masks (we can skip this now. This part needs more time)
```

### Citation
If you use our code or models in your research, please cite with:

```
@inproceedings{wan2025robust,
  title={Robust Zero-Shot Crowd Counting and Localization With Adaptive Resolution SAM},
  author={Wan, Jia and Wu, Qiangqiang and Lin, Wei and Chan, Antoni},
  booktitle={European Conference on Computer Vision},
  pages={478--495},
  year={2025},
  organization={Springer}
}
```

