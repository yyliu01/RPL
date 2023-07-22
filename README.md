# RPL
> [Residual Pattern Learning for Pixel-wise Out-of-Distribution Detection in Semantic Segmentation](https://arxiv.org/pdf/2211.14512.pdf)
>
> by Yuyuan Liu*, Choubo Ding*, [Yu Tian](https://yutianyt.com/), [Guansong Pang](https://sites.google.com/site/gspangsite),
> [Vasileios Belagiannis](https://campar.in.tum.de/Main/VasileiosBelagiannis), 
> [Ian Reid](https://cs.adelaide.edu.au/~ianr/) and [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/)
> 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/residual-pattern-learning-for-pixel-wise-out/anomaly-detection-on-fishyscapes-1)](https://paperswithcode.com/sota/anomaly-detection-on-fishyscapes-1?p=residual-pattern-learning-for-pixel-wise-out)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/residual-pattern-learning-for-pixel-wise-out/anomaly-detection-on-road-anomaly)](https://paperswithcode.com/sota/anomaly-detection-on-road-anomaly?p=residual-pattern-learning-for-pixel-wise-out)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/residual-pattern-learning-for-pixel-wise-out/anomaly-detection-on-fishyscapes-l-f)](https://paperswithcode.com/sota/anomaly-detection-on-fishyscapes-l-f?p=residual-pattern-learning-for-pixel-wise-out)

<img src="https://github.com/yyliu01/RPL/assets/102338056/4307ce17-9c44-4e19-82b7-b0508f51ff28.png" width="700" height="300" />

### Installation
please install the dependencies and dataset based on this [***installation***](./docs/installation.md) document.

### Getting start
please follow this [***instruction***](./docs/before_start.md) document to reproduce our results.

### Results
our training logs and checkpoints are in this [***result***](./docs/result.md) page.

## Acknowledgement & Citation 

Our code is highly based on the [PEBAL](https://github.com/tianyu0207/PEBAL). 
Please consider citing them in your publications if they help your research.
```bibtex
@article{liu2022residual,
  title={Residual Pattern Learning for Pixel-wise Out-of-Distribution Detection in Semantic Segmentation},
  author={Liu, Yuyuan and Ding, Choubo and Tian, Yu and Pang, Guansong and Belagiannis, Vasileios and Reid, Ian and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2211.14512},
  year={2022}
}
@inproceedings{tian2022pixel,
  title={Pixel-wise energy-biased abstention learning for anomaly segmentation on complex urban driving scenes},
  author={Tian, Yu and Liu, Yuyuan and Pang, Guansong and Liu, Fengbei and Chen, Yuanhong and Carneiro, Gustavo},
  booktitle={European Conference on Computer Vision},
  pages={246--263},
  year={2022},
  organization={Springer}
}
```
#### TODO
- [x] RPL code has been released.
- [x] RPL+CoroCL code has been released.
- [x] The results based on extra training sets (e.g., Vistas, Wilddash2) have been released.
