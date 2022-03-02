# Overview
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

# Abstract
Hair editing is an interesting and challenging problem in computer vision and graphics. Many existing methods require well-drawn sketches or masks as conditional inputs for editing, however these interactions are neither straightforward nor efficient. In order to free users from the tedious interaction process, this paper proposes a new hair editing interaction mode, which enables manipulating hair attributes individually or jointly based on the texts or reference images provided by users. For this purpose, we encode the image and text conditions in a shared embedding space and propose a unified hair editing framework by leveraging the powerful image text representation capability of the Contrastive Language-Image Pre-Training (CLIP) model. With the carefully designed network structures and loss functions, our framework can perform high-quality hair editing in a disentangled manner. Extensive experiments demonstrate the superiority of our approach in terms of manipulation accuracy, visual realism of editing results, and irrelevant attribute preservation.

# Comparison
## Comparison to Text-Driven Image Manipulation Methods
<img src='assets/comparison-text.png'>

## Comparison to Hair Transfer Methods
<img src='assets/comparison-image.png'>

# Application
## Hair Interpolation
<img src='assets/interpolation.png'>

## Generalization Ability to Unseen Descriptions
<img src='assets/generalization.png'>

## Cross-Modal Conditional Inputs
<img src='assets/cross-modal .png'>

# To Do
- [ ] Release testing code
- [ ] Release pretrained model
- [ ] Release training code

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
