# Pytorch Intro

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Generic badge](https://img.shields.io/badge/Model-passing-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Plotting-passing-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/dataset-passing-green.svg)](https://shields.io/)

Building LeNet-5 by pytorch and use MNIST dataset.

## Structure

- Basic tutorial-1: [mnist_tutorial-1](./notebook/mnist_tutorial.ipynb)
- Basic tutorial-2: [mnist_tutorial-2](./notebook/Pytorch_tutorial_02_Improve_Neural_Networks.md)
- Power point: [ppt](./notebook/Mnist_Tutorial.pptx)

```text
pytorch_intro/
├── main.py
├── model.py
├── notebook
│   ├── mnist_tutorial.ipynb
│   ├── Mnist Tutorial.pptx
│   ├── Pytorch tutorial 01_ MNIST and Pretrained Model.md
│   └── Pytorch tutorial 02_ Improve Neural Networks.md
├── readme.md
├── requirements.txt
├── tree.txt
└── utils.py
```

## RUN

**Clone repo:**

```bash
https://github.com/maglili/pytorch_intro.git
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