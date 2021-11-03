# Pytorch Intro

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Generic badge](https://img.shields.io/badge/Model-passing-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Plotting-passing-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/dataset-passing-green.svg)](https://shields.io/)

Building LeNet-5 by pytorch and use MNIST dataset.
[**Jupyter notebook version**](./resources/mnist_tutorial.ipynb) and
[**PPT**](./resources/MNIST_TUTORIAL.pdf) are available in `resources/` folder.

## Structure

```text
pytorch_intro/
├── resources
├── submission
├── data
├── main.py
├── model.py
└── utils.py
```

## Requirement

1. pytorch
2. matplotlib
3. torchsummary
4. tqdm

*Requirements is not recommended:* You should check your GPU for install proper version of Pytorch.

```bash
#pip install -r requirements.txt # this may arise package error
```

## RUN

**Clone repo:**

```bash
https://github.com/cosbi-nckuee/lab-training.git
```

**Parameters detail:**

```bash
python main.py -h
```

**Training:**

```bash
python main.py -m train -bs 64 -epo 4 -lr 5e-4
```

**Predict:**

```bash
python main.py -m pred -bs 64
```

## Reference

1. [maglili/dqn-pytorch](https://github.com/maglili/dqn-pytorch)
2. [maglili/ai_cup-movie_comment](https://github.com/maglili/ai_cup-movie_comment)
