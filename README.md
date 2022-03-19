# HairCLIP: Design Your Hair by Text and Reference Image (CVPR2022)
> This repository hosts the official PyTorch implementation of the paper: "**HairCLIP: Design Your Hair by Text and Reference Image**".

Our **single** framework supports **hairstyle and hair color editing** individually or jointly, and conditional inputs can come from either **image** or **text** domain. 

<img src='assets/teaser.png'>


Tianyi Wei<sup>1</sup>,
Dongdong Chen<sup>2</sup>,
Wenbo Zhou<sup>1</sup>,
Jing Liao<sup>3</sup>,
Zhentao Tan<sup>1</sup>,
Lu Yuan<sup>2</sup>, 
Weiming Zhang<sup>1</sup>, 
Nenghai Yu<sup>1</sup> <br>
<sup>1</sup>University of Science and Technology of China, <sup>2</sup>Microsoft Cloud AI, <sup>3</sup>City University of Hong Kong

## News
**`2022.03.02`**: Our paper is accepted by CVPR2022 and the code will be released soon.   
**`2022.03.09`**: Our training code is released.  
**`2022.03.19`**: Our testing code and pretrained model are released. 

## Getting Started
### Pretrained Models
Please download the pre-trained model from the following link. The HairCLIP model contains the entire architecture, including the mapper and decoder weights.
| Path | Description
| :--- | :----------
|[HairCLIP](https://drive.google.com/file/d/1tGIKWI6xniEULeyADvhu_xzRVcv_31Sx/view?usp=sharing)  | Our pre-trained HairCLIP model.

If you wish to use the pretrained model for training or inference, you may do so using the flag `--checkpoint_path`.  
In addition, we provide various auxiliary models and latent codes inverted by [e4e](https://github.com/omertov/encoder4editing) needed for training your own HairCLIP model from scratch.
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1pts5tkfAcWrg4TpLDu6ILF5wHID32Nzm/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1FS2V756j-4kWduGxfir55cMni5mZvBTv/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during HairCLIP training.
|[CelebA-HQ Train Set Latent Codes](https://drive.google.com/file/d/1gof8kYc_gDLUT4wQlmUdAtPnQIlCO26q/view?usp=sharing) | Train set latent codes inverted by [e4e](https://github.com/omertov/encoder4editing).
|[CelebA-HQ Test Set Latent Codes](https://drive.google.com/file/d/1j7RIfmrCoisxx3t-r-KC02Qc8barBecr/view?usp=sharing) | Test set latent codes inverted by [e4e](https://github.com/omertov/encoder4editing).

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`.

# To Do
- [x] Release training code
- [x] Release testing code
- [x] Release pretrained model

## Citation

If you find our work useful for your research, please consider citing the following papers :)

```
@article{wei2022hairclip,
  title={Hairclip: Design your hair by text and reference image},
  author={Wei, Tianyi and Chen, Dongdong and Zhou, Wenbo and Liao, Jing and Tan, Zhentao and Yuan, Lu and Zhang, Weiming and Yu, Nenghai},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
