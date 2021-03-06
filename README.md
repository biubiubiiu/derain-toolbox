# Derain-toolbox

This is a single image deraining toolbox based on [PyTorch](https://github.com/pytorch/pytorch) and [mmcv](https://github.com/open-mmlab/mmcv). Some benchmark algorithms for single image deraining are reimplemented in this repo, which are migrated from their official code or reproduced based on their papers (for those have no publicly available code). Experiments of these algorithms are carried out again, including evaluation on benchmark datasets (Rain200L, Rain200H, Rain800, Rain1200, and Rain1400) and tests of network complexity. The results are reported in the corresponding README documents.

## Model Zoo

- [DerainNet](configs/derainnet/README.md) (TIP' 2017) \[[paper](https://ieeexplore.ieee.org/abstract/document/7893758/)\] \[[code](https://xueyangfu.github.io/projects/tip2017.html)\]
- [DDN](configs/ddn/README.md) (CVPR' 2017) \[[paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Fu_Removing_Rain_From_CVPR_2017_paper.html)\] \[[code](https://xueyangfu.github.io/projects/cvpr2017.html)\]
- [RESCAN](configs/rescan/README.md) (ECCV' 2018) [[paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Xia_Li_Recurrent_Squeeze-and-Excitation_Context_ECCV_2018_paper.html)] \[[code](https://github.com/XiaLiPKU/RESCAN)\]
- [PReNet](configs/prenet/README.md) (CVPR' 2019) \[[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Ren_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline_CVPR_2019_paper.html)\] \[[code](https://github.com/csdwren/PReNet)\]
- [ReHEN](configs/rehen/README.md) (ACMMM' 2019) \[[paper](https://dl.acm.org/doi/10.1145/3343031.3351149)\] \[[code](https://github.com/nnUyi/ReHEN)\]
- [LPNet](configs/lpnet/README.md) (TNNLS' 2019) \[[paper](https://ieeexplore.ieee.org/document/8767931)\] \[[code](https://xueyangfu.github.io/projects/LPNet.html)\]
- [Physical Model Guided ID](configs/physical_model_guided/README.md) (ICME' 2020) \[[paper](https://www.computer.org/csdl/proceedings-article/icme/2020/09102878/1kwr8NheVtm)\] \[[code](https://github.com/Ohraincu/PHYSICAL-MODEL-GUIDED-DEEP-IMAGE-DERAINING)\]
- [MARDNet](configs/mardnet/README.md) (ACCV' 2020) \[[paper](https://openaccess.thecvf.com/content/ACCV2020/html/Chen_Multi-scale_Attentive_Residual_Dense_Network_for_Single_Image_Rain_Removal_ACCV_2020_paper.html)\] \[[code](https://github.com/cxtalk/MARD-Net)\]
- [OUCDNet](configs/oucdnet/README.md) (JSTSP' 2020) \[[paper](https://ieeexplore.ieee.org/abstract/document/9264746)\] \[[code](https://github.com/jeya-maria-jose/Derain_OUCD_Net)\]
- [ID-CGAN](configs/cgan/README.md) (TCSVT' 2020) \[[paper](https://ieeexplore.ieee.org/document/8727938)\] \[[code](https://github.com/hezhangsprinter/ID-CGAN)\]
- [DCSFN](configs/dcsfn/README.md) (ACMMM' 2020) \[[paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413820)\] \[[code](https://github.com/Ohraincu/DCSFN)]
- [DRD-Net](configs/drdnet/README.md) (CVPR' 2020) \[[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_Detail-recovery_Image_Deraining_via_Context_Aggregation_Networks_CVPR_2020_paper.html)\] \[[code](https://github.com/Dengsgithub/DRD-Net)\]
- [RCDNet](configs/rcdnet/README.md) (CVPR' 2020) \[[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.html)\] \[[code](https://github.com/hongwang01/RCDNet_simple)\]
- [DualGCN](configs/dual_gcn/README.md) (AAAI' 2021) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16224)\] \[[code](https://xueyangfu.github.io/paper/2021/AAAI/code.zip)\]
- [SPDNet](configs/spdnet/README.md) (ICCV' 2021) \[[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yi_Structure-Preserving_Deraining_With_Residue_Channel_Prior_Guidance_ICCV_2021_paper.html)\] \[[code](https://github.com/Joyies/SPDNet)\]
- [DerainRLNet](configs/rlnet/README.md) (CVPR' 2021) \[[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.html)\] \[[code](https://github.com/LI-Hao-SJTU/DerainRLNet)\]
- [ECNet](configs/ecnet/README.md) (WACV' 2022) \[[paper](https://openaccess.thecvf.com/content/WACV2022/html/Li_Single_Image_Deraining_Network_With_Rain_Embedding_Consistency_and_Layered_WACV_2022_paper.html)\] \[[code](https://github.com/Yizhou-Li-CV/ECNet)\]

## Installation

See [installation instructions](docs/installation.md)

## Getting Started

See [getting started](docs/getting_started.md) for the basic usage of derain-toolbox.

## Credits

Portions of this project utilized other open-source works, the use of which is listed acknowledged in [CREDITS](CREDITS.md).

## Contributions

If you've found an error in this project, please file an issue.

Patches are encouraged and may be submitted by forking this project and
submitting a pull request. Since this project is still in its very early stages,
if your change is substantial, please raise an issue first to discuss it.

## License

```
Copyright 2021 Raymond Wong

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
