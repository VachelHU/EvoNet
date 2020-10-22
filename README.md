# EvoNet
This project implements the Evolutionary State Graph Neural Network proposed in [1], which is a GNN-based method for time-series event prediction.

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

[1] Wenjie, H; Yang, Y; Ziqiang, C; Carl, Y and Xiang, R, 2021, Time-Series Event Prediction with Evolutionary State Graph, In WSDM, 2021
```
@inproceedings{hu2021evonet, 
    title={Time-Series Event Prediction with Evolutionary State Graph},
    author={Wenjie Hu and Yang Yang and Ziqiang Cheng and Carl Yang and Xiang Ren},
    booktitle={Proceedings of WSDM},
    year={2021}
}
```

