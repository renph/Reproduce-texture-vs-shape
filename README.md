# ICLR2019-Reproducibility-Challenge

Paper: [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://openreview.net/forum?id=Bygh9j09KX)
Code: [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape)
1. 确定重复的主要论点，找出文中依据与实验数据
1. 总结文章的逻辑思路


---
## abstract
论文定量地研究了 CNNs 在识别物体时，图像的纹理和形状如何影响决策。
作者发现，基于ImageNet训练的CNN非常喜欢依据纹理做决策，而不是形状，这与人类的行为相反。
同时，作者阐释了相同的模型（ResNet-50）可以学习基于纹理的特征表示（使用ImageNet数据），也可以学习基于形状的特征表示（使用stylized ImageNet数据）。
这样得到的模型更接近人类的行为，并且有意想不到的好处。例如，提高在物体识别中的表现，对多种图像扭曲有前所未见的鲁棒性，突出了形状表示具有优越性。

## introduction
CNN可以有效识别物体。一个直觉上的解释是CNN不断组合低级的特征（如边缘）得到复杂形状（如车轮、车窗），直到物体（如汽车）可以恰好与其他类区别开。
【例子1，2】
这种解释称之为形状假说。

这个假设有大量的经验证据支持。【例子1，2，3，4】

另外，其他的研究表明物体的纹理对CNN识别物体也很重要。【例子1，2，3，4，5】
总之，似乎局部的纹理真的提供了足够的物体类别信息。
这意味着，ImageNet的分类 原则上可以只使用纹理识别来完成。
作者相信，使用CNN识别物体时，物体的纹理比全局的形状更重要。
这是纹理假说。

解决这两个互相矛盾的假说对理解CNN意义很大。
这项工作中，作者设计了直观的实验来解决这个问题。
使用风格迁移可以得到形状和纹理相矛盾的图案，用来定量分析人和CNN在物体识别中的bias。
实验结果表明，纹理假说更有可能正确。
此外，作者另外两个贡献是changing biases（例如，把CNN原本基于纹理决策改为基于形状决策）和探索changing biases的好处。
作者发现如果使用合适的数据集，可以把CNN原本基于纹理决策改为基于形状决策。主要基于形状决策的CNN更加鲁棒。

## methods
### 心理学实验
pass
### 数据集
为了评估texture biases 和 shape biases，作者做了6个主实验和3个控制实验。
前五个实验是简单的物体识别（如图2所示），改变原图，再尝试识别。
注意，实验使用的原图是所有4个CNN都能正确识别的。
这是为了与第6个实验对照，方便解释结果。
第6个实验使用纹理与形状矛盾的图片，要求CNN尝试分类。
只有原图是所有4个CNN都能正确识别的，才能保证该实验的重心在验证纹理假说或形状假说。
该试验要求人类被试保持绝对中立态度（即不要偏向形状或纹理）。
因为没有正确的答案，作者关注被试的主观想法。

### stylized-ImageNet （SIN）
基于ImageNet得到的新数据集。去除图像的纹理，并用随机的风格替代（使用AdaIN，如图3所示）。
## results
### TEXTURE VS SHAPE BIAS IN HUMANS AND IMAGENET-TRAINED CNNS

### OVERCOMING THE TEXTURE BIAS OF CNNS

### ROBUSTNESS AND ACCURACY OF SHAPE-BASED REPRESENTATIONS

## discussion

## conclusion
作者证实今天的机器识别基本上依赖物体的纹理而不是大部分人认同的形状。
作者展示了使用基于形状表示决策的鲁棒性和好处。
作者预见他们的发现和他们开放的模型和代码将达成3个目标：
1. 对CNN的特征表示和偏差（bias）有更深入的理解
1. 向获得更合理的人类视觉物体识别模型迈出了一步。 
1. 这是未来工作的起点，其中的领域知识（domain knowledge）表明基于形状的表示可能比基于纹理的表示更有用。