from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
import pandas
# load data
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['dur',	'proto',	'service',	'state'	,'spkts'	,'dpkts'	,'sbytes'	,'dbytes'	,'rate'	,'sttl'	,'dttl'	,'sload'	,'dload'	,'sloss',	'dloss',	'sinpkt',	'dinpkt',	'sjit',	'djit',	'swin',	'stcpb',	'dtcpb',	'dwin',	'tcprtt',	'synack', 'ackdat',	'smean',	'dmean',	'trans_depth',	'response_body_len',	'ct_srv_src',	'ct_state_ttl',	'ct_dst_ltm',	'ct_src_dport_ltm',	'ct_dst_sport_ltm',	'ct_dst_src_ltm',	'is_ftp_login',	'ct_ftp_cmd',	'ct_flw_http_mthd',	'ct_src_ltm',	'ct_srv_dst',	'is_sm_ips_ports',	'label']
df = pandas.read_csv("Concatenation.csv", names=names, dtype=object)
array = df.values

cat_df_flights_onehot = df.copy()
cat_df_flights_onehot = pandas.get_dummies(cat_df_flights_onehot, columns=['dur'], prefix = ['dur'])
print(cat_df_flights_onehot.head())
"""
X = array[:,0:42]
Y = array[:,42]
print(df.head())
print(df.info())

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
"""