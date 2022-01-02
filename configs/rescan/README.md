# RESCAN (ECCV'2018)

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_ECCV_2018/html/Xia_Li_Recurrent_Squeeze-and-Excitation_Context_ECCV_2018_paper.html">Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining (ECCV'2018)</a></summary>

```bibtex
@inproceedings{li2018recurrent,
  title={Recurrent squeeze-and-excitation context aggregation net for single image deraining},
  author={Li, Xia and Wu, Jianlong and Lin, Zhouchen and Liu, Hong and Zha, Hongbin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={254--269},
  year={2018}
}
```

</details>

<br/>

&nbsp;

**Quantitative Result**

The metrics are `PSNR/SSIM`. Both are evaluated on RGB channels.

> **_NOTE:_** Following the authors' setup, random seed is set to 66 in all experiments.

|                                Method                                |  Rain200L   |  Rain200H   |   Rain800   |  Rain1200   |  Rain1400   |
| :------------------------------------------------------------------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|   [RESCAN(ConvRNN+Add)](/configs/rescan/rescan_c24s4d5_rnn_add.py)   | 36.00/0.973 | 25.75/0.813 | 25.98/0.838 | 31.87/0.897 | 30.62/0.904 |
|  [RESCAN(ConvRNN+Full)](/configs/rescan/rescan_c24s4d5_rnn_full.py)  | 35.93/0.972 | 25.75/0.803 | 26.01/0.825 | 31.86/0.895 | 30.64/0.904 |
|   [RESCAN(ConvGRU+Add)](/configs/rescan/rescan_c24s4d5_gru_add.py)   | 36.84/0.977 | 26.35/0.830 | 26.37/0.844 | 32.23/0.904 | 30.88/0.909 |
|  [RESCAN(ConvGRU+Full)](/configs/rescan/rescan_c24s4d5_gru_full.py)  | 36.74/0.977 | 26.65/0.836 | 26.42/0.838 | 32.17/0.903 | 30.87/0.909 |
|  [RESCAN(ConvLSTM+Add)](/configs/rescan/rescan_c24s4d5_lstm_add.py)  | 36.85/0.977 | 26.25/0.827 | 26.57/0.847 | 32.17/0.903 | 30.91/0.909 |
| [RESCAN(ConvLSTM+Full)](/configs/rescan/rescan_c24s4d5_lstm_full.py) | 36.91/0.977 | 26.50/0.833 | 26.58/0.842 | 32.23/0.904 | 31.02/0.911 |

&nbsp;

**Network Complexity**

|      Method      |  Input shape  |    Flops    | Params  |
| :--------------: | :-----------: | :---------: | :-----: |
| RESCAN(ConvGRU)  | (3, 250, 250) | 30.97GFlops | 150.22k |
| RESCAN(ConvLSTM) | (3, 250, 250) | 41.23GFlops | 197.76k |
| RESCAN(ConvRNN)  | (3, 250, 250) | 11.80GFlops | 55.13k  |
