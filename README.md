# Code and Experiments for "Universality of AdaGrad Stepsizes for Stochastic Optimization" 

[![ci](https://github.com/mlolab/universal-adagrad-experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/mlolab/universal-adagrad-experiments/actions/workflows/ci.yml)

This repository contains the official implementation of the experiments
presented in our NeurIPS 2024 paper, *Universality of AdaGrad Stepsizes for
Stochastic Optimization: Inexact Oracle, Acceleration and Variance Reduction*
([PDF](https://openreview.net/pdf?id=rniiAVjHi5)).

## Overview

The implementation of the methods proposed in our paper can be found under
[`src/universal_adagrad/optimizers`](https://github.com/mlolab/universal-adagrad-experiments/tree/main/src/universal_adagrad/optimizers).
These include:

- **UniSgd**
- **UniFastSgd**
- **UniSvrg**
- **UniFastSvrg**

Additionally, we provide implementations of the following baseline methods:

- [SVRG with constant stepsize](http://proceedings.mlr.press/v48/allen-zhub16.pdf)
- [Accelerated SVRG with constant stepsize (a primal version of the VRADA method)](https://proceedings.neurips.cc/paper_files/paper/2020/file/093b60fd0557804c8ba0cbf1453da22f-Paper.pdf)
- [AdaSVRG](https://arxiv.org/pdf/2102.09645.pdf)
- [AdaVRAE](https://proceedings.mlr.press/v162/liu22o/liu22o.pdf)
- [AdaVRAG](https://proceedings.mlr.press/v162/liu22o/liu22o.pdf)


## Dependencies
The package [Poetry](https://python-poetry.org) is required to be installed on
your system. For detailed guidance, please follow the [official installation
instructions](https://python-poetry.org/docs/#installation).

## Installation
1)  Clone the repository and navigate to the directory.
    ```shell
    git clone git@github.com:mlolab/universal-adagrad-experiments.git
    cd universal-adagrad-experiments
    ```

2) Install dependencies defined in `pyproject.toml`:
   ```shell
   poetry install
   ```

3) Test installation (optional):
   ```shell
   poetry run pytest
   ```
   
## Running Experiments

Run the following command to execute all experiments and generate the
corresponding figures presented in the paper:

```shell
poetry run reproduce-experiments
```

The process may take several hours depending on your system's resources, so
please be patient.

By default, the results and figures are saved in the `results` directory within
the current working directory.
This can be altered by using the `--savedir-base` flag.
If the script is interrupted, whether intentionally or accidentally, it will
automatically resume from where it left off when we re-run, avoiding redundant
computations and preserving previously completed results.

## Citation

If you find this code useful for your research, please consider citing our
paper. Below is the BibTeX code for your convenience:

```bibtex
@article{rodomanov2024universality,
  title = {
     {Universality} of {AdaGrad Stepsizes} for {Stochastic Optimization}:
     {Inexact Oracle}, {Acceleration} and {Variance Reduction}
  },
  author = {Rodomanov, Anton and Jiang, Xiaowen and Stich, Sebastian},
  journal = {arXiv preprint arXiv:2406.06398},
  year = {2024}
}
```
