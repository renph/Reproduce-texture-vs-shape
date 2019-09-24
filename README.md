# Reproduce-texture-vs-shape

Paper: [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://openreview.net/forum?id=Bygh9j09KX)

Repo: [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape)

## Introduction
We reproduced an oral ICLR paper, named ``ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness''. We implemented the most important part in the original paper and briefly discussed experiments on a subset of the ImageNet dataset (only 16 classes). 
We found that ResNet-50 trained on Stylized ImageNet is more accurate and robust than the same network trained only on ImageNet. We also verified that shape-based representations are more robust than the texture representations.
All codes except the style transfer are written by the two authors.

## Dataset
A subset of ImageNet (only 16 classes), referred as IN-16, and stylized IN-16, referred as SIN-16, are used as raw input for training.
The figure below shows an example of stylized images.
![Stylized Dataset](report/img/bear.png)

Datasets can be downloaded from [imagenet-16](https://www.kaggle.com/davidddxu/imagenet16) 
and [stylizedimagenet-16](https://www.kaggle.com/davidddxu/stylizedimagenet16).

Code for style transfer using AdaIN can be found here ([stylize-datasets](https://github.com/Hvitgar/stylize-datasets)).

## Texture Bias

## Results
### Performance

### Robustness

## Conclusion
