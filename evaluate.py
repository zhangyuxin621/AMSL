import csv
import numpy as np

csv_file = csv.reader(open(r'train_normal_loss_sum_mse.csv', 'r'))
data1=[]
for row in csv_file:
    data1.append(row)
b_t = []
for i in range(len(data1)):
    b_t.append(float(data1[i][0]))
bb_t=np.array(b_t)
thre = np.percentile(bb_t, 98)  #Percentile

csv_file = csv.reader(open(r'normal_loss_sum_mse', 'r'))
data=[]
for row in csv_file:
    data.append(row)
b = []
for i in range(len(data)):
    b.append(float(data[i][0]))
bb=np.array(b)
print(bb.shape)

normal_true = bb[bb[:] < thre]
print('normal_sum :' ,bb.shape[0],'normal_true',normal_true.shape[0])

csv_file = csv.reader(open(r'abnormal_loss_sum_mse', 'r'))
data=[]
for row in csv_file:
    data.append(row)
b = []
for i in range(len(data)):
    b.append(float(data[i][0]))
bb1=np.array(b)
print(bb1.shape)

abnormal_true = bb1[bb1[:] >= thre]
print('abnormal_sum :' ,bb1.shape[0],'normal_true',abnormal_true.shape[0])

normal_true = normal_true.shape[0]
fatigue_true = abnormal_true.shape[0]
recerr2 = bb.shape[0]
recerr3 = bb1.shape[0]

acc = (normal_true + fatigue_true) / (recerr2 + recerr3)
precision_n = normal_true / (normal_true + recerr3 - fatigue_true)
precision_a = fatigue_true / (fatigue_true + recerr2 - normal_true)
pre_avg = (precision_a + precision_n) / 2
recall_n = normal_true / recerr2
recall_a = fatigue_true / recerr3
recall_avg = (recall_a + recall_n) / 2
F1 = 2 * (pre_avg * recall_avg) / (pre_avg + recall_avg)

from prettytable import PrettyTable
x = PrettyTable(["acc", "pre_normal", "pre_abn", "pre_avg", "recall_normal", "recall_abn", "recall_avg", "F1"])
x.add_row([acc, precision_n, precision_a, pre_avg, recall_n, recall_a, recall_avg, F1])
print(x)