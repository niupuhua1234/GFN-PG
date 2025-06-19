  <a href="https://www.python.org/"><img alt="Python Version" src="https://img.shields.io/badge/Python-%E2%89%A53.10-blue" /></a>
  <a href="https://pytorch.org/"><img alt="PyTorch Version" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0.0-green" /></a>
<!-- <div align="center">
</div> -->
--------------------------------------------------------------------------------
# GFN-PG
Code for the ICML 2024 paper '[GFlowNet Training by Policy Gradients](https://proceedings.mlr.press/v235/niu24c.html)'

[Clik here for downloading sEH dataset](https://drive.google.com/file/d/1-RCgm7CZFmU3UP4pw8CXI9r_LiUaUSP4/view?usp=sharing)

The code is adapted from [*torchgfn*](https://pypi.org/project/torchgfn/0.1.3/) but not compatible with it. Please make sure *torchgfn* is not installed in your Python environment when running the code, in case of some unexpected function importing. 

## Installation
Clone the code and set ```./GFN-PG/gfn-pg``` as the root working directory

Create environment with conda:
```
conda create -n gfn_env python==3.10
conda activate gfn_env

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirement.txt
```



## Citation

If you find our code useful, please considering citing our paper in your publications. We provide a BibTeX entry below.


```bibtex
@inproceedings{niu2024gflownet,
  title={GFlowNet Training by Policy Gradients},
  author={Niu, Puhua and Wu, Shili and Fan, Mingzhou and Qian, Xiaoning},
  booktitle={International Conference on Machine Learning},
  pages={38344--38380},
  year={2024},
  organization={PMLR}
}


