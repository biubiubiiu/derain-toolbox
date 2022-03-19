# RLNet (CVPR'2021)

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.html">Robust Representation Learning With Feedback for Single Image Deraining (CVPR'2021)</a></summary>

```bibtex
@inproceedings{chen2021robust,
  title={Robust representation learning with feedback for single image deraining},
  author={Chen, Chenghao and Li, Hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7742--7751},
  year={2021}
}
```

</details>

<br/>

![rlnet](../../figs/rlnet.png)

<br/>

**Quantitative Result**

The metrics are `PSNR/SSIM`. Both are evaluated on RGB channels.

> **_NOTE:_**
>
> - Number of training epochs is reduced to 24 for Rain1200 and Rain1400.
> - The official code and the paper are somehow inconsistent:
> 
>    | Paper                                                                                                                                                                                                     | Official Code                                                                                                                                                                                           |
>    |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
>    | Train in two stages                                                                                                                                                                                       | End-to-end training                                                                                                                                                                                     |
>   | Batch size is 1                                                                                                                                                                                         | Batch size is 2                                                                                                                                                                                         |
>   | Training epochs: 90 + 240                                                                                                                                                                               | Training epochs: 242                                                                                                                                                                                    |
>    | <img src="https://render.githubusercontent.com/render/math?math={%5Ctheta1%3D0.15%2C%5Ctheta_2%3D0.05%20(%5Ctext%7Bin%20stage%202%7D)}&mode=inline">                                                                                                | <img src="https://render.githubusercontent.com/render/math?math={%5Ctheta1%3D0.05%2C%5Ctheta_2%3D0.15}&mode=inline">                                                                                              |
>    | <img src="https://render.githubusercontent.com/render/math?math={%5Clambda%3D0.01(%5Ctext%7Bin%20stage%202%7D)}&mode=inline">                                                                                                | <img src="https://render.githubusercontent.com/render/math?math={%5Clambda%3D0.006}&mode=inline">                                                                                              |
>    | <img src="https://render.githubusercontent.com/render/math?math={%5Clambda_2%3D0%5C%20%5Ctext%7Bwhen%20reach%7D%5C%2030%7B%5Ctimes%7DK%5Ctext%7Bepochs%7D(K%3D1%2C2%2C3...)}&mode=inline">                | <img src="https://render.githubusercontent.com/render/math?math={%5Clambda_2%3D0%5C%20%5Ctext%7Bwhen%20reach%7D%5C%2030%7B%5Ctimes%7DK%5Ctext%7Bepochs%7D(K%3D0%2C1%2C2...)}&mode=inline">              |
>    | <img src="https://render.githubusercontent.com/render/math?math={%5Clambda_2%3D0.6%5C%20%5Ctext%7Bwhen%20reach%7D%5C%2030%7B%5Ctimes%7DK%2B15%5C%20%5Ctext%7Bepochs%7D(K%3D1%2C2%2C3...)}&mode=inline"> | <img src="https://render.githubusercontent.com/render/math?math={%5Clambda_2%3D0.6%5C%20%5Ctext%7Bwhen%20reach%7D%5C%2030%7B%5Ctimes%7DK%2B15%5C%20%5Ctext%7Bepochs%7D(K%3D0%2C1%2C2...)}&mode=inline"> |
> - Three sciripts are provided for reproducing this work: [rlnet_official_code.py](./rlnet_official_code.py) is directly migrated from the official code, while [rlnet_stage1.py](./rlnet_stage1.py) and [rlnet_stage2.py](./rlnet_stage2.py) follows the settings in the paper.


|                            Method                             |  Rain200L   |  Rain200H   |   Rain800   |  Rain1200   |  Rain1400   |
| :-----------------------------------------------------------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| [RLNet(official code)](/configs/rlnet/rlnet_official_code.py) | 36.56/0.978 | 27.67/0.874 | 27.19/0.866 | 32.37/0.914 | 30.72/0.916 |
|        [RLNet(paper)](/configs/rlnet/rlnet_stage2.py)         | -----/----- | -----/----- | -----/----- | -----/----- | -----/----- |

<br/>

**Network Complexity**

|  Input shape  |    Flops    | Params |
| :-----------: | :---------: | :----: |
| (3, 256, 256) | 68.94GFlops | 5.82M  |
