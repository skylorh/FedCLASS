# FedCLASS

This repository is a Pytorch implementation of **FedCLASS**:

[**Federated Class-Incremental Learning with New-Class Augmented Self-Distillation**.](https://arxiv.org/abs/2312.11489) *arXiv preprint arXiv:2401.00622*. 2024.

Contains implementations of FedAvg [1], GLFC [2], FedAvg+WA [3], FedAvg+ICARL [4] as described in the paper.

-------
## Contributions
- We propose FedCLASS, a novel FCIL method that mitigates catastrophic forgetting by harmonizing new class scores with the outputs of historical models during selfdistillation.
- We provide theoretical analyses for FedCLASS that conform to the soundness of FedCLASS’s design. **To our best knowledge, FedCLASS is the first federated class-incremental learning method with theoretical support.**
- We conduct extensive empirical experiments on four datasets with two class-incremental settings. Results demonstrate that FedCLASS substantially reduces the average forgetting rate and markedly enhances global accuracy compared with state-of-the-art methods.

-------

## Running

```python ./fl_trainer.py --config＝[config_name].json```

## Cite this work

```bibtex
@article{wu2024federated,
  title={Federated Class-Incremental Learning with New-Class Augmented Self-Distillation},
  author={Wu, Zhiyuan and He, Tianliu and Sun, Sheng and Wang, Yuwei and Liu, Min and Gao, Bo and Jiang, Xuefeng},
  journal={arXiv preprint arXiv:2401.00622},
  year={2024}
}
```

## References

[1] McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. "Communication-efficient learning of deep networks from decentralized data." In *Artificial intelligence and statistics*, pp. 1273-1282. PMLR, 2017.

[2] Dong, Jiahua, Lixu Wang, Zhen Fang, Gan Sun, Shichao Xu, Xiao Wang, and Qi Zhu. "Federated class-incremental learning." In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 10164-10173. 2022.

[3] Zhao, Bowen, Xi Xiao, Guojun Gan, Bin Zhang, and Shu-Tao Xia. "Maintaining discrimination and fairness in class incremental learning." In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 13208-13217. 2020.

[4] Rebuffi, Sylvestre-Alvise, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. "icarl: Incremental classifier and representation learning." In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*, pp. 2001-2010. 2017.