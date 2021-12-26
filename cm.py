import numpy as np
from scipy.misc import logsumexp
import pandas
# cm = np.array(
# [[      0,       0,     210,       0,      58,       0,       0,       0,       0, 0],
#  [      0,       0,     199,       1,      30,       3,       0,       0,       0, 0],
#  [      0,       0,    9010,      60,    2203,      57,      13,      10,       0, 0],
#  [      0,       0,   15495,     152,   23445,     179,     229,      25,       0, 0],
#  [      0,     0 ,   6946,      18,   11956,      65,     200,      61,       0, 0],
#  [      0,    0  ,207374,      18,    2938,      85,      55,      11,       0, 0],
#  [      0,     0   , 8710,      36,   30292,     503, 2147555,     360,       0, 0],
#  [      0,    0 ,4673,      14,    4292,       6,       1,       1,       0, 0],
#  [      0,   0     , 80,       0 ,     72,       0,       0,       0,       0, 0],
#  [      0 ,  0,       5,       0,      13,       0,       0,       0,       0, 0]])
cm = np.array(
[[      15,       1,     180,       4,      49,       2,        3,          5,        7,     2],
 [      1,       121,     60,       5,      30,       3,        5,          4,        1,     3],
 [      2,       9,    10077,      360,    703,      407,       941,        50,       3,     2],
 [      0,       0,     5495,     13152,   9001,     323,       11529,      25,       0,     0],
 [      1,       2,      497,      318,   18807,     262,       256,        103,      0,     0],
 [      0,       0,    97374,      1018,    2938,    100085,    9055,       11,       0,     0],
 [      0,       0,     8710,      36,      30292,   503,       2147555,    358,      2,     0],
 [      0,       0,     2673,      1014,    2292,     6,         1,         3001,     0,     0],
 [      0,       0,     60,         3 ,     72,       0,         13,         0,       4,     0],
 [      0,      0,       5,         0,      11,       0,          0,         0,       0,     2]])

"""

analysis : training samples = 2409.0 | total samples = 2677
backdoor : training samples = 2096.0 | total samples = 2329
dos : training samples = 5000 | total samples = 16353
exploits : training samples = 5000 | total samples = 44525
fuzzers : training samples = 5000 | total samples = 24246
generic : training samples = 5000 | total samples = 215481
normal : training samples = 31000 | total samples = 2218456
reconnaissance : training samples = 5000 | total samples = 13987
shellcode : training samples = 1359.0 | total samples = 1511
worms : training samples = 156.0 | total samples = 174



Number of categories is 9 | Total samples in categories:
| 0: 2677
|1: 2329
|2: 16353
|3: 44525
|4: 24246
|5: 215481
|6: 13987
|7: 1511
|8: 174
normal samples for rf: 31000
normal samples for nn: 5000



 Analysis | Backdoor | DoS   | Exploits | Fuzzers | Generic | Normal  | Reconnaissance | Shellcode | Worms 
The number of attacks of type Exploits: 44525
The number of attacks of type Reconnaissance: 13987
The number of attacks of type DoS: 16353
The number of attacks of type Generic: 58871
The number of attacks of type Shellcode: 1511
The number of attacks of type Worms: 174
The number of attacks of type Backdoors: 2329
The number of attacks of type Analysis: 2677
The number of attacks of type Fuzzers: 24246
The number of data points of type Normal:93000


"""



FP = cm.sum(axis=0)-np.diag(cm)  
FN = cm.sum(axis=1)-np.diag(cm)
TP = np.diag(cm)
TN = cm.sum()-(FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
FPR = FP/(FP+TN)
print("False Alarm Rate is")
print(FPR)

print()
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()
def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows
def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns




print("Confusion Matrix \n")
print ("          0A       1A       2A       3A       4A       5A       6A       7A       8A       9A")
row_labels = ['0P', '1P', '2P', '3P','4P','5P','6P','7P','8P','9P']

for row_label, row in zip(row_labels, cm):
    print('%s [%s]' % (row_label, ' '.join('%08s' % i for i in row)))

print("Label   Precision   Recall   FAR")
for label in range(10):
    print(f"{label:2d} {precision(label, cm):15.9f} {recall(label, cm):13.9f}")

print("\nPrecision total: ", precision_macro_average(cm))
print("\nRecall total: ", recall_macro_average(cm))
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    print("\n The accuracy is \n")
    print(diagonal_sum / sum_of_all_elements) 

accuracy(cm)
