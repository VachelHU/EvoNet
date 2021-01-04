# EvoNet
This project implements the Evolutionary State Graph Neural Network proposed in [1], which is a GNN-based method for time-series event prediction.

## Compatibility

Code is compatible with tensorflow version 1.2.0 and Pyhton 3.6.2.

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

### Input Format 

An example data format is given where data is stored as a list containing 4 dimensionals tensors such as
 
`[number of samples × segment number × segment length × dimension of observation]`


### Configuration
We can use `./model_core/config.py` to set the parameters of model.

```
class ModelParam(object):
    # basic
    model_save_path = "./model"
    n_jobs = os.cpu_count()

    # dataset
    data_path = './data'
    data_name = 'webtraffic'
    his_len = 15
    segment_len = 24
    segment_dim = 2
    n_event = 2
    norm = True

    # state recognition
    n_state = 30
    covariance_type = 'diag'

    # model
    graph_dim = 256
    node_dim = 96
    learning_rate = 0.001
    batch_size = 1000
    id_gpu = '0'
    pos_weight = 1.0
```


### Main Script

```
python run.py -h

usage: run.py [-h] [-d {djia30, webtraffic}] [-g GPU]

optional arguments:
  -h, --help            show this help message and exit
  -d {djia30,webtraffic}, --dataset {djia30,webtraffic} select the dataset
  -g GPU, --gpu GPU     target gpu id
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

