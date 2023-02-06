
# Results
*we claim that we utilise **ONE** single model to perform all the evaluations to simulate driving scenarios in real life.*

### cityscapes reports

| methods 	| mIoU (class) |
|:--------:	|:-----:	|
|closed seg. **[[paper](https://arxiv.org/pdf/1812.01593.pdf)]** **[[code](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet)]**| 82.46 **[[report](https://www.cityscapes-dataset.com/anonymous-results/?id=59c2112456c9ccfa371b9d07c14ad1af459a49c0c1dcb1b0f86206696724c7f7)]**	||
|dense-hybrid **[[paper](https://arxiv.org/pdf/2207.02606.pdf)]** **[[code](https://github.com/matejgrcic/DenseHybrid)]**| 82.06 **[[report](https://www.cityscapes-dataset.com/anonymous-results/?id=0d7223c203dc2437367c61314ee055b8e7b7eb7f9b1667e330eec3855956b063)]**	| |
|meta-ood **[[paper](https://arxiv.org/pdf/2012.06575.pdf)]** **[[code](https://github.com/robin-chan/meta-ood)]**| 81.51 **[[report](https://www.cityscapes-dataset.com/anonymous-results/?id=d34a108dc2b75f03eaac6bd790c631dbd8448c93a3a18ad870dc2102669dca8c)]**	|  |
|pebal **[[paper](https://arxiv.org/pdf/2111.12264.pdf)]** **[[code](https://github.com/tianyu0207/PEBAL)]**| 81.19 **[[report](https://www.cityscapes-dataset.com/anonymous-results/?id=bdaa6c69751b6a1cfe0c08db66f4ba96967a161124873e0a8fb6afbaf01f3098)]**	|  |
|**ours** | **82.46 [[report](https://www.cityscapes-dataset.com/anonymous-results/?id=e51cc63dc225379f5a974d54cec04d6a4135481446d0180c3a950bb7d96d8c4c)]**	| |

### checkpoints
you can reproduce our results based on the supported checkpoints below:
* **rpl** can download in **[here](https://drive.google.com/drive/folders/1XHyvdT2LJzbzVJyoNOUHVtReKGg6HkLq?usp=share_link)**, and **rpl+corocl** can download in **[here](https://drive.google.com/drive/folders/1rVaBRdOpS2JkAo-ZRO64jSZU0VbdZsDn?usp=share_link)**.

for the segment-me-if-you-can (SMIYC), download the official **[evaluation code](https://github.com/adynathos/road-anomaly-benchmark)** (with an extra post-process stage) to achieve the reported performance. we support the prediction & results in [here](https://drive.google.com/drive/folders/1oE9CQCyvdE-Jt6akNE3wFpnk_q6ueONs?usp=share_link).
### training details

1) you can download our **training log** via this **[LINK](https://drive.google.com/drive/folders/1Ba3IpT4CY5hxvGkvBfHLNIcD89Ml8Hmm?usp=share_link)**.
2) for more details, you can check our wandb log in this
   **[LINK](https://wandb.ai/yy/RPL?workspace=user-pyedog1976)**,
   where it includes:
   1) <img src="https://user-images.githubusercontent.com/102338056/167979073-1c1b3144-8a72-4d8d-9084-31d7fdab3e9b.png" width="26" height="22"> overall information (e.g., training command line, hardware information and training time).
   2) <img src="https://user-images.githubusercontent.com/102338056/167978940-8c1f3d79-d062-4e7b-b56e-30b97d273ae8.png" width="26" height="22"> training details (e.g., loss curves, validation results and visualization)
   3) <img src="https://user-images.githubusercontent.com/102338056/167979238-4847430f-aa0b-483d-b735-8a10b43293a1.png" width="26" height="22"> output logs (well, sometimes might crash ...)
* in the final training stage, we adopt longer training epochs and more frequent validation to choose the potential best model for **black boxing test sets**.
