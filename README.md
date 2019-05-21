# ESGRN
This project implements the Evolutionary State Graph Recurrent Networks proposed in [1], which is a GNN-based method for time series modeling.

## Compatibility

Code is compatible with tensorflow version 1.1.0 and Pyhton 3.6.2.

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
  -d {earthquake,webtraffic}, --dataset {earthquake,webtraffic}
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

[1] Wenjie Hu, Yang Yang, Zilong You, Zongtao Liu, Xiang Ren, 2019, Modeling Combinatorial Evolution in Time Series Prediction, In [arXiv:1905.05006v2](https://arxiv.org/abs/1905.05006v2), 2019