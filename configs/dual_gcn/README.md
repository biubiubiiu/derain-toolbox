# DualGCN (AAAI'2021)

<details>
<summary align="right"><a href="https://ojs.aaai.org/index.php/AAAI/article/view/16224">Rain Streak Removal via Dual Graph Convolutional Network</a></summary>

```bibtex
@inproceedings{fu2021rain,
  title={Rain streak removal via dual graph convolutional network},
  author={Fu, Xueyang and Qi, Qi and Zha, Zheng-Jun and Zhu, Yurui and Ding, Xinghao},
  booktitle={Proc. AAAI Conf. Artif. Intell.},
  pages={1--9},
  year={2021}
}
```

</details>

<br/>

![dual_gcn](../../figs/dual_gcn.png)

<br/>

**Quantitative Result**

The metrics are `PSNR/SSIM`. Both are evaluated on RGB channels.

> **_NOTE:_**
>
> - Number of training epochs is reduced to 30 for Rain1200 and Rain1400.

|                         Method                          |  Rain200L   |  Rain200H   |   Rain800   |  Rain1200   |  Rain1400   |
| :-----------------------------------------------------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| [dual_gcn_c72n11](/configs/dual_gcn/dual_gcn_c72n11.py) | 36.84/0.979 | 27.07/0.860 | 23.68/0.799 | 30.83/0.882 | 30.27/0.904 |

<br/>

**Network Complexity**

|  Input shape  |    Flops     | Params  |
| :-----------: | :----------: | :-----: |
| (3, 250, 250) | 169.54GFlops | 845.82k |