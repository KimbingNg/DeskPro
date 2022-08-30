# Identity-Sensitive Knowledge Propagation for Cloth-Changing Person Re-identification

This is the official implementation of the paper "Identity-Sensitive Knowledge Propagation for Cloth-Changing Person Re-identification"

[![Paper](https://img.shields.io/badge/arXiv-2208.12023-important)](https://arxiv.org/pdf/2208.12023)

![deskpro](https://user-images.githubusercontent.com/50580578/186648071-9f4264bc-9a3c-48d9-beb3-d654432347c6.png)

## Getting Started

```sh
git clone https://github.com/KimbingNg/DeskPro && cd DeskPro
```
Download the prepared datasets and the pretrained teacher models from [this link](https://drive.google.com/drive/folders/1_Q3UqOP3eEhOUR06u6a3vOI29Xbq0efP), and runs
```
tar -zxvf dataset.tar.gz
pip3 install -r ./requirements.txt
```

**Train on Celeb**:
```sh
dataset=celeb
CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./config_files/celeb_config.yaml \
 data_test $dataset"_lr_hr_mask" data_train $dataset"_lr_hr_mask" \
 loss '1*CrossEntropy+1*Triplet' \
 tag 'exp_version' \
 batchid 5 \
 batchimage 6 \
 kd_loss.enable True \
 kd_loss.T 5. \
 kd_loss.alpha 0.7 \
 mse.mse_weight 7.0 \
 mse.spatial_attn_lr 1.0 \
 forward_mode all \
 pre_train "$dataset"_teacher.pth
```


**Train on Celeb-light**:
```sh
dataset=celeb-light
CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./config_files/celeb_config.yaml \
 data_test $dataset"_lr_hr_mask" data_train $dataset"_lr_hr_mask" \
 loss '1*CrossEntropy+1*Triplet' \
 tag 'exp_version' \
 batchid 5 \
 batchimage 6 \
 kd_loss.enable True \
 kd_loss.T 5. \
 kd_loss.alpha 0.7 \
 mse.mse_weight 7.0 \
 mse.spatial_attn_lr 1.0 \
 forward_mode all \
 pre_train "$dataset"_teacher.pth
```

**Train on PRCC**:
```sh
dataset=prcc
CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./config_files/prcc_config.yaml \
 data_test $dataset"_lr_hr_mask" data_train $dataset"_lr_hr_mask" \
 loss '1*CrossEntropy+1*Triplet' \
 tag 'exp_version' \
 batchid 5 \
 batchimage 6 \
 kd_loss.enable True \
 kd_loss.T 1. \
 kd_loss.alpha 0.8 \
 mse.mse_weight 7.0 \
 mse.spatial_attn_lr 1.0 \
 forward_mode all \
 pre_train "$dataset"_teacher.pth
```

## Citation:

If you find this work useful in your research, please consider citing:

```
@inproceedings{wuIdentitySensitiveKnowledgePropagation2022,
  title = {Identity-{{Sensitive Knowledge Propagation}} for {{Cloth-Changing Person Re-identification}}},
  booktitle = {2022 {{IEEE International Conference}} on {{Image Processing}}},
  author = {Wu, Jianbing and Liu, Hong and Shi, Wei and Tang, Hao and Guo, Jingwen},
  year = {2022},
  publisher = {{IEEE}}
}
```


### Acknowledgments
The codes was built on top of  [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid), [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline), [MGN-pytorch](https://github.com/seathiefwang/MGN-pytorch), and [LightMBN](https://github.com/jixunbo/LightMBN), we thank the authors for sharing their code publicly.
