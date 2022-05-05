# **CycleGAN-VC3-PyTorch**

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/jackaduma/CycleGAN-VC2)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/jackaduma?locale.x=zh_XC)

[**中文说明**](./README.zh-CN.md) | [**English**](./README.md)

------

This code is a **PyTorch** implementation for paper: [CycleGAN-VC3: Examining and Improving CycleGAN-VCs for Mel-spectrogram Conversion](https://arxiv.org/abs/2010.11672]), a nice work on **Voice-Conversion/Voice Cloning**.

- [x] Dataset
  - [ ] VC
- [x] Usage
  - [x] Training
  - [x] Example 
- [ ] Demo
- [x] Reference

------

## **CycleGAN-VC3**

### [**Project Page**](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc3/index.html) 


Non-parallel voice conversion (VC) is a technique for learning mappings between source and target speeches without using a parallel corpus. Recently, CycleGAN-VC [3] and CycleGAN-VC2 [2] have shown promising results regarding this problem and have been widely used as benchmark methods. However, owing to the ambiguity of the effectiveness of CycleGAN-VC/VC2 for **mel-spectrogram conversion**, they are typically used for mel-cepstrum conversion even when comparative methods employ mel-spectrogram as a conversion target. To address this, we examined the applicability of CycleGAN-VC/VC2 to **mel-spectrogram conversion**. Through initial experiments, we discovered that their direct applications compromised the time-frequency structure that should be preserved during conversion. To remedy this, we propose CycleGAN-VC3, an improvement of CycleGAN-VC2 that incorporates **time-frequency adaptive normalization (TFAN)**. Using TFAN, we can adjust the scale and bias of the converted features while reflecting the time-frequency structure of the source mel-spectrogram. We evaluated CycleGAN-VC3 on inter-gender and intra-gender non-parallel VC. A subjective evaluation of naturalness and similarity showed that for every VC pair, CycleGAN-VC3 outperforms or is competitive with the two types of CycleGAN-VC2, one of which was applied to mel-cepstrum and the other to mel-spectrogram.

![network comparison](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc3/images/comparison.png "comparison between vc2 and vc3")  _Figure 1. We developed time-frequency adaptive normalization (TFAN), which extends instance normalization [5] so that the affine parameters become element-dependent and are determined according to an entire input mel-spectrogram._

------

**This repository contains:** 

1. [TFAN module code](tfan_module.py) which implemented the TFAN module
1. [model code](model.py) which implemented the model network.
2. [audio preprocessing script](preprocess_training.py) you can use to create cache for [training data](data).
3. [training scripts](train.py) to train the model.



------

## **Table of Contents**

- [**CycleGAN-VC3-PyTorch**](#cyclegan-vc3-pytorch)
  - [**CycleGAN-VC3**](#cyclegan-vc3)
    - [**Project Page**](#project-page)
  - [**Table of Contents**](#table-of-contents)
  - [**Requirement**](#requirement)
  - [**Usage**](#usage)
  - [**Star-History**](#star-history)
  - [**Reference**](#reference)
  - [Donation](#donation)
  - [**License**](#license)
  
------

## **Requirement** 

```bash
pip install -r requirements.txt
```
## **Usage**


------

## **Star-History**

![star-history](https://api.star-history.com/svg?repos=jackaduma/CycleGAN-VC3&type=Date "star-history")

------

## **Reference**
1. **CycleGAN-VC3: Examining and Improving CycleGAN-VCs for Mel-spectrogram Conversion.** [Paper](https://arxiv.org/abs/2010.11672), [Project](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc3/index.html)
2. CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion. [Paper](https://arxiv.org/abs/1904.04631), [Project](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html)
3. Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks. [Paper](https://arxiv.org/abs/1711.11293), [Project](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/)
4. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. [Paper](https://arxiv.org/abs/1703.10593), [Project](https://junyanz.github.io/CycleGAN/), [Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
5. Image-to-Image Translation with Conditional Adversarial Nets. [Paper](https://arxiv.org/abs/1611.07004), [Project](https://phillipi.github.io/pix2pix/), [Code](https://github.com/phillipi/pix2pix)


------

## Donation
If this project help you reduce time to develop, you can give me a cup of coffee :) 

AliPay(支付宝)
<div align="center">
	<img src="./misc/ali_pay.png" alt="ali_pay" width="400" />
</div>

WechatPay(微信)
<div align="center">
    <img src="./misc/wechat_pay.png" alt="wechat_pay" width="400" />
</div>

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://paypal.me/jackaduma?locale.x=zh_XC)


------

## **License**

[MIT](LICENSE) © Kun