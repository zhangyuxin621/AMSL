# AMSL

Code implementation for : [Adaptive Memory Networks with Self-supervised Learning for Unsupervised Anomaly Detection(TKDE 2021)](https://arxiv.org/pdf/2201.00464.pdf)


# Installation
### Requirements
* Python == 3.6
* Cuda == 9.1
* Keras ==2.2.2
* Tensorflow ==1.8.0


# Usage
We use DASADS dataset(refer to [UCI](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities)) as demo example. 

## Data Preparation
```
# run preprocessing.py to generate normal and abnormal datasets.

data
 |-DASADS
 | |-a01  
 | | |-p1  
 | | | |-s01.txt
 | |-...
 | |-a09
 |-generate_dataset
 | |-normal.npy
 | |-abnormal.npy

 # run transformation.py to transform all the data and then separate them into training and testing datasets.

 data
 |-generate_dataset
 | |-normal.npy
 | |-abnormal.npy
 |-transform_dataset
 | |-train_dataset
 | | |-data_raw_train.npy  # raw data
 | | |-data_no_train.npy   # noise data
 | | |-data_ne_train.npy   # negated data
 | | |-data_op_train.npy   # opposite_time data
 | | |-data_pe_train.npy   # permuted data
 | | |-data_sc_train.npy   # scale data
 | | |-data_ti_train.npy   # time_warp data
 | |-test_dataset
 | | |-normal data
 | | | |-data_raw_test.npy  # raw data
 | | | |-data_no_test.npy   # noise data
 | | | |-data_ne_test.npy   # negated data
 | | | |-data_op_test.npy   # opposite_time data
 | | | |-data_pe_test.npy   # permuted data
 | | | |-data_sc_test.npy   # scale data
 | | | |-data_ti_test.npy   # time_warp data
 | | |-abnormal data
 | | | |-data_raw_abnormal.npy  # raw data
 | | | |-data_no_abnormal.npy   # noise data
 | | | |-data_ne_abnormal.npy   # negated data
 | | | |-data_op_abnormal.npy   # opposite_time data
 | | | |-data_pe_abnormal.npy   # permuted data
 | | | |-data_sc_abnormal.npy   # scale data
 | | | |-data_ti_abnormal.npy   # time_warp data

```

## Run

### Train model

You can get results of the MSE loss after running train.py. 

```

results
 |-train_normal_loss_sum_mse.csv  #the MSE loss of training data
 |-normal_loss_sum_mse.csv  #the MSE loss of normal data in the testing dataset
 |-abnormal_loss_sum_mse.csv #the MSE loss of abnormal data in the testing dataset

```

### Evaluation

Run evaluate.py to compute the threshold by the MSE loss of training data and achieve the accuracy, precision, recall and F1 score of testing data.

## Paper introduction

Unsupervised anomaly detection aims to build models to effectively detect unseen anomalies by only training on the normal data. Although previous reconstruction-based methods have made fruitful progress, their generalization ability is limited due to two critical challenges. First, the training dataset only contains normal patterns, which limits the model generalization ability. Second, the
feature representations learned by existing models often lack representativeness which hampers the ability to preserve the diversity of normal patterns (see Fig.1). In this paper, we propose a novel approach called Adaptive Memory Network with Self-supervised Learning (AMSL) to address these challenges and enhance the generalization ability in unsupervised anomaly detection. Based on the convolutional autoencoder structure, AMSL incorporates a self-supervised learning module to learn general normal patterns and an adaptive memory fusion module to learn rich feature representations. Experiments on four public multivariate time series datasets demonstrate that AMSL significantly improves the performance compared to other state-of-the-art methods. Specifically, on the largest CAP sleep stage detection dataset with 900 million samples, AMSL outperforms the second-best baseline by 4%+ in both accuracy and F1 score.
Apart from the enhanced generalization ability, AMSL is also more robust against input noise.

![1|center](./picture2/6.png)

### Proposed Approach

![1|center](./picture2/5.png)
\centerline{Fig.2: The structure of the proposed AMSL.}

We propose a novel Adaptive Memory Network with Self-supervised Learning (AMSL) for unsupervised anomaly detection. AMSL consists of four novel components as shown in Fig. 2: 1) a self-supervised learning module, 2) a global memory module, 3) a local memory module and 4) an adaptive fusion module.

### Comparison Methods
![1|center](./picture2/4.png)
\centerline{TABLE 3 The comparison of mean precision, recall, F1 and accuracy of AMSL and other baselines.}

TABLE 3 reports the overall performance results on these public datasets. It can be observed that the proposed AMSL method achieves significantly superior performance over the baseline methods in all the datasets. Specifically, compared with other methods, AMSL significantly improves the F1 score by 9.07% on PAMAP2 dataset, 4.90% on CAP
dataset, 8.77% on DSADS dataset and 2.35% on WESAD dataset. The same pattern goes for precision and recall. Especially for the largest CAP dataset with over 900 Millon
samples, AMSL dramatically outperforms the second-best baseline (OCSVM) with an F1 score of 4.90%, indicating its effectiveness.
