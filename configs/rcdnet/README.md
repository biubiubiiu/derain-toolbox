# RCDNet (CVPR'2020)

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.html">A Model-Driven Deep Neural Network for Single Image Rain Removal (CVPR'2020)</a></summary>

```bibtex
@inproceedings{wang2020model,
  title={A model-driven deep neural network for single image rain removal},
  author={Wang, Hong and Xie, Qi and Zhao, Qian and Meng, Deyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3103--3112},
  year={2020}
}
```

</details>

<br/>

![rcdnet](../../figs/rcdnet.png)

<br/>

**Quantitative Result**

The metrics are `PSNR/SSIM`. Both are evaluated on RGB channels.

|                        Method                         |  Rain200L   |  Rain200H   |   Rain800   |  Rain1200   |  Rain1400   |
| :---------------------------------------------------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| [rcdnet_c32s17n4](/configs/rcdnet/rcdnet_c32s17n4.py) | 38.66/0.985 | 28.73/0.885 | 27.46/0.867 | 32.24/0.908 | 31.02/0.914 |

<br/>

**Network Complexity**

|  Input shape  |    Flops     | Params |
| :-----------: | :----------: | :----: |
| (3, 256, 256) | 194.54GFlops | 2.97M  |
