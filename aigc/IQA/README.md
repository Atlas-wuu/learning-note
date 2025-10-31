# 图像质量评估（IQA）
## 数据集
### KADID-10k
**数据集简介：** 

KADID-10k数据集由Konstanz大学提供，从Pixabay.com收集分辨率大于1500×1200的原始图像，被重新缩放到与TID2013 (512×384)相同的分辨率，同时保持像素纵横比，若需要即进行裁剪。最后，手动选择81张高质量的图像作为KADID-10k的原始图像，每张原始图像进行5种程度的25种失真退化。

**数据格式：**
- 数据集包括81组图像，每组图像包括原图及25种不同类型的模拟损伤。

- 标注文件 dmos.csv 包含了对应图片与原图之间的dmos分，以及各标注结果的方差信息。

- 标注文件 val.csv 包含了各图像模糊、压缩、噪声的模拟损伤程度，可用于评估图像画质损伤。

**下载链接：** https://www.modelscope.cn/datasets/iic/KADID-10k-database

### TID2013
**数据集简介：** 

TID2013数据集包含了从25个参考图中获得的3000多张测试图像，每个参考图像有24种失真类型，每种类型的失真分为5 个级别。通过对来自五个不同国家（芬兰、法国、意大利、乌克兰和美国）的志愿者进行了985次主观实验，收集了图像的平均意见得分


**下载链接：** https://www.ponomarenko.info/tid2013.htm

### KonIQ-10k
**数据集简介：** 

KonIQ-10k 数据集是一个生态有效的大规模图像质量评估数据集，包含了 10,000 张自然图像，每张图像都经过多人评分，提供了丰富的标注信息。这个数据集的多样性和高质量标注使其成为训练 IQA 模型的理想选择。

**下载链接：** https://osf.io/hcsdy/files

## 算法
### aesthetic-predictor-v2-5-2024
**介绍：** 基于SigLIP的预期器， 评分范围从1-10

**github：** https://github.com/discus0434/aesthetic-predictor-v2-5

### improved-aesthetic-predictor-2022
**介绍：** 基于CLIP+MLP进行图像质量评估，

**github：** https://github.com/christophschuhmann/improved-aesthetic-predictor

### simple-aesthetics-predictor-2023
**介绍：** 基于CLIP进行图像质量评估

**github：** https://github.com/shunk031/simple-aesthetics-predictor

### VisualQuality-R1-NeurIPS 2025
**介绍：** 基于Qwen2.5-VL-7B-Instruct模型在3个数据集KADID-10K, TID2013, KonIQ-10k上finetuen而来

**github：** https://github.com/TianheWu/VisualQuality-R1?tab=readme-ov-file

