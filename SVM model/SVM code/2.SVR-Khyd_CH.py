import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import sys
import csv


pattern=list()
fhd=csv.reader(open('K_hyd_CH data.csv','r'))
for line in fhd:
    pattern.append(line)

#raw_sample
raw_sample=list()
fhd=csv.reader(open('K_hyd_CH predict.csv','r'))
for line in fhd:
    raw_sample.append(line)

#label
label=list()
fhl=csv.reader(open('K_hyd_CH Standard.csv','r'))
for line in fhl:
    label.append(line)

label=np.array(label,dtype='float64')
label=label.tolist()


predict = list()
fhp = csv.reader(open('K_hyd_CH predict.csv','r'))
for line in fhd:
    predict.append(line)

predict=np.array(predict,dtype='float64')
predict=predict.tolist()

#SVR
raw_train=list()
label_train=list()
raw_test=list()
label_test=list()

i,j,v=1,1,1
for line in pattern:
    if i==4:
        raw_test.append(line)
        i=1
    else:
        raw_train.append(line)
        i=i+1
for line in label:
    if j==4:
        label_test.append(line)
        j=1
    else:
        label_train.append(line)
        j=j+1



clf=svm.SVR(kernel='linear') #kernel:linear、rbf、poly
#clf = RandomForestRegressor(random_state=1)
clf.fit(raw_train,label_train)
label_predict=clf.predict(raw_test)
label_predict=label_predict.tolist()
label_predict2 = clf.predict(raw_sample)
label_predict2=label_predict2.tolist()
output1 =csv.writer(open('label_predict.csv','a',newline=''),dialect='excel')
output2 =csv.writer(open('label_predict2.csv','a',newline=''),dialect='excel')
output1.writerows(map(lambda x:[x],label_predict))
output2.writerows(map(lambda x:[x],label_predict2))
