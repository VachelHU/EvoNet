# EvoNet
This project implements the Evolutionary State Graph Neural Network proposed in [1], which is a GNN-based method for time series analysis.

## Compatibility

Code is compatible with tensorflow version 1.2.0 and Pyhton 3.6.2.

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

### Input Format 

An example data format is given where data is stored as a list containing 4 dimensionals tensors such as [number of samples × segment number × segment length × dimensionality].

### Main Script

```
python run.py -h

usage: run.py [-h] [-v STATENUM] [-d {earthquake,webtraffic}]
              [-lr LEARNING_RATE] [-b BATCHSIZE] [-g GPU] [-p MODELPATH]

optional arguments:
  -h, --help            show this help message and exit
  -v STATENUM, --statenum STATENUM
                        state number
  -d {djia30,webtraffic}, --dataset {djia30,webtraffic}
                        select the dataset
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate
  -b BATCHSIZE, --batchsize BATCHSIZE
                        batch size
  -g GPU, --gpu GPU     state number
  -p MODELPATH, --modelpath MODELPATH
                        the path of storing model
```

## Reference

[1] Wenjie Hu, Yang Yang, Ziqiang Cheng, Carl Yang, Xiang Ren, 2020, Modeling Evolutionary State Graph for Time Series Prediction
, In [arXiv:1905.05006v2](https://arxiv.org/abs/1905.05006v2), 2020

```
@article{hu2019evolutionarygraph,
  author    = {Wenjie Hu, Yang Yang, Ziqiang Cheng, Carl Yang, Xiang Ren},
  title     = {Modeling Evolutionary State Graph for Time Series Prediction},
  journal   = {CoRR},
  volume    = {abs/1905.05006},
  year      = {2020},
  url       = {https://arxiv.org/abs/1905.05006},
  archivePrefix = {arXiv},
  eprint    = {1905.05006},
}
```

