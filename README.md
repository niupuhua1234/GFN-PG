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
```


## Technical chronology and originality/attribution concern

Our work, *GFlowNet Training by Policy Gradients*, was submitted to ICLR 2024 on September 23, 2023, made publicly available on OpenReview on October 13, 2023, and subsequently accepted at ICML 2024.

We have documented a detailed technical comparison between the Soft-TRPO formulation introduced in *GFlowNet Training by Policy Gradients*(2024), and the Ent-PPO formulation presented in the subsequent work *Proximal Policy Optimization for Amortized Discrete Sampling*(2026). In particular, the comparison shows how the Ent-PPO objective can be derived by applying standard PPO approximation techniques to the earlier Soft-TRPO formulation.  The comparison also examines the corresponding value and advantage definitions, Soft-RL interpretation, derivations of performance-difference and performance-bound for policy optimization at the equation level.

Based on these publicly documented technical correspondences, we consider the presentation of previously established conceptual, methodological, and theoretical results as new contributions in *Proximal Policy Optimization for Amortized Discrete Sampling* to constitute **serious academic misconduct concerning originality and attribution**, rather than merely a matter of missing citations or incomplete related-work discussion.

The complete technical evidence and chronology, based solely on publicly available versions of the respective works, are documented in the following records:

- **[Full technical comparison and evidence](TECHNICAL_COMPARISON.md)**
- **[Original public record on OpenReview](https://openreview.net/forum?id=k2lkeCCfRK&noteId=R9ob5DJqcf)**

