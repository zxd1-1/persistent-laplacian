import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pylab import *
import copy
dirlst=sorted(os.listdir(r'D:\rois_aal\Outputs\cpac\nofilt_noglobal\rois_aal'), key=str.lower)
dirlst1=sorted(os.listdir(r'D:\rois_aal_spectral_matrix_features'),key=str.lower)
feature_matrix=np.zeros((884,312))
for item in dirlst1:
    a=np.loadtxt(r'D:\rois_aal_spectral_matrix_features\{0}'.format(item))
    feature_matrix[dirlst1.index(item)]=a[:312]
column_means=np.mean(feature_matrix,axis=0)
for i in range(312):
    for j in range(884):
        feature_matrix[j][i]=feature_matrix[j][i]-column_means[i]
#np.savetxt('spectral_feature_matrix_312',feature_matrix,fmt='%.2e')
feature_matrix=feature_matrix.tolist()

data=pd.DataFrame(feature_matrix,index=["subject_"+str(i+1) for i in range(884)])
print(data)
lst=[]
for i in range(312):
    if data.iloc[0,i]==0:
        lst.append(i)
data.drop(data.iloc[:,lst],axis=1,inplace=True)
new_columns = [i for i in range(1, len(data.columns) + 1)]
data.columns = new_columns
#print(data)
#data.to_excel(r'312aalmeanfeaturedata.xlsx',index=False)
df=pd.read_excel(r"C:\Users\Administrator\Desktop\Persistent-Laplacian-Method-main (1)\Persistent-Laplacian-Method-main\Persistent Laplacian Method\Persistent Laplacian Method\Code\data.xlsx")
lst_new=[]

df_new=pd.DataFrame({'Unamed':[],'Unamed':[],'SITE_ID':[],'FILE_ID':[],
                    'DSM_IV_TR':[],'AGE_AT_SCAN':[] ,'SEX':[],'FIQ':[],'VIQ':[],
                     'PIQ':[],'HANDEDNESS_CATEGORY':[],'DX_GROUP':[]})
#print(df_new)
#num=0
for item in df['FILE_ID']:
    n=df[df['FILE_ID'] == item].index
    if r'{0}_rois_aal.1D.txt'.format(item) in dirlst1:
        #print(n)
        row_to_add=df.iloc[n]
        df_new=pd.concat([df_new,pd.DataFrame(row_to_add)],axis=0,ignore_index=True)
df_new.drop(df_new.iloc[:,[0]],axis=1,inplace=True)
#print(df_new)
df_new_sorted=df_new.sort_values(by='FILE_ID' ,key=lambda x: x.str.lower())
df_new_sorted.rename(columns={'Unnamed: 0':'Number', 'Unnamed: 1':'Center','SITE_ID':'Siteid',
                              'FILE_ID':'fileid','DSM_IV_TR':'DSMTR','AGE_AT_SCAN':'AGE',
                              'HANDEDNESS_CATEGORY':'HANDEDNESS','DX_GROUP':'DXGROUP'}, inplace=True)
df_new_sorted = df_new_sorted.loc[:,['Number','Center','Siteid','fileid','DSMTR','AGE','SEX','FIQ','VIQ',
                     'PIQ','HANDEDNESS','DXGROUP']]
#print(df_new_sorted)
#df_new_sorted.to_excel(r'884data.xlsx',index=False)
txt=np.loadtxt(r"D:\程序\intensity_combat_eyes_TR_site_new312.txt")

#print(len(txt))
combat_feature_matrix=np.zeros((884,312))
nonzero_column_means=[]
for i in column_means:
    if i !=0:
        nonzero_column_means.append(i)
#print(len(nonzero_column_means))
for i in range(312):
    for j in range(884):
        combat_feature_matrix[j][i]=txt[j][i]+nonzero_column_means[i]
np.savetxt('884312combat_feature_matrix',combat_feature_matrix)
combat_feature_matrix=np.loadtxt('884312combat_feature_matrix')
dataset = pd.DataFrame(combat_feature_matrix)
data=pd.read_excel('884data.xlsx')
Y1= data['AGE']
#print(Y1)
Y=[]
for i in range(884):
    Y.append(Y1[i])
mean_Y=np.mean(Y)
#for column in dataset.columns:
lst1,lst2=[],[]
for i in range(312):
    X=[]
    for j in range(884):
         X.append(dataset[i][j])

    up=0
    down1=0
    for k in range(884):
        up+=X[k]*(Y[k]-mean_Y)
        down1+=(Y[k]**2)

    w=up/(down1-(((np.sum(Y))**2)/884))
    b=0
    for l in range(884):
        b+= (X[l]-w*Y[l])

    lst1.append(w)
    lst2.append(b/884)
# print(lst1)
# print(lst2)
residual_numpy=np.zeros((884,312))
residual=residual_numpy.tolist()
for i in range(312):
    for j in range(884):
        residual[j][i]=dataset[i][j]-lst1[i]*Y1[j]-lst2[i]
#print(np.asarray(residual))
np.savetxt('884312residual_matrix和年龄回归',np.asarray(residual))
matrix111=np.loadtxt('884312residual_matrix和年龄回归')

dirlst2=sorted(os.listdir(r'D:\rois_aal_coor_matrix'),key=str.lower)
feature1=np.zeros((884,6670))
for item in dirlst2[:1]:
    lst=[]
    a=np.loadtxt(r'D:\rois_aal_coor_matrix\{0}'.format(item))
    #print(a)
    for i in range(116):
        for j in range(i+1,116):
            lst.append(a[i][j])
    feature1[dirlst2.index(item)]=np.array(lst)



class_feature_matrix=np.zeros((884,312))
ar=np.zeros(884)
import random
number=list(range(884))
random.shuffle(number)
for i in range(884):
    class_feature_matrix[i]=matrix111[number[i]]#[:312]只有正特征
    ar[i]=data['DXGROUP'][number[i]]
    if ar[i]==2:
        ar[i]=0
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold

curr_score=0
spe_score=0
sen_score=0
a=0
#feature_importance=list(np.zeros(87))
kf=KFold(n_splits=10,shuffle=True,random_state=0)
# print(kf)
feature_importance=list(np.zeros(312))
data,target=class_feature_matrix[:] ,ar[:]
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j]=round(data[i][j],2)
data=np.array(data)
kf=KFold(n_splits=10,shuffle=True,random_state=0)
curr_score=0
spe_score=0
sen_score=0
curr_score1=0
f1_score=0
for train_index,test_index in kf.split(data):
    clt2 = RandomForestClassifier(max_depth=4,n_estimators=75,random_state=42,criterion='entropy').fit(data[train_index], target[train_index])#max_depth=15,n_estimators=1000,random_state=42
    curr_score = curr_score + clt2.score(data[test_index], target[test_index])
    curr_score1 = curr_score1 + clt2.score(data[train_index], target[train_index])
    print('准确率为：',clt2.score(data[test_index],target[test_index]))
    for i in range(312):
        feature_importance[i]+=clt2.feature_importances_[i]
    target_exchange=[]
    predict_exchange=[]
    for item in target[test_index]:
        if item== 2:
            item=0
            target_exchange.append(item)
        else:
            target_exchange.append(item)
    for item in clt2.predict_proba(list(data[test_index])):
            predict_exchange.append(item[0])
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(list(data[test_index]))):
        if list(clt2.predict(list(data[test_index])))[i] == 1:
            if list(target[test_index])[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if list(target[test_index])[i] == 1:
                fn += 1
            else:
                tn += 1
    tnr = tn / (fp + tn+0.001)
    tpr = tp / (tp + fn+0.001)
    spe_score += tnr
    sen_score += tpr
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1score = (2 * recall * precision) / (recall + precision)
    f1_score += F1score
    print('specificity为：', tnr)
    print('sensitivity为：', tpr)
    print('F1-score为:', F1score)
avg_score2=curr_score/10
avg_spe2=spe_score/10
avg_sen2=sen_score/10
avg_f1score=f1_score/10
print('平均准确率为：',avg_score2)
print('平均specificity为：',avg_spe2)
print('平均sensitivity为:',avg_sen2)
print('平均F1score为：',avg_f1score)


mean_feature_importance=list(np.zeros(312))
for i in range(312):
    mean_feature_importance[i]=feature_importance[i]/10
#print(mean_feature_importance)
copy1=copy.deepcopy(mean_feature_importance)
copy1.sort()
d=copy1[-120:]
print(d)
lst0=[]
for i in d:
    lst0.append((mean_feature_importance.index(i)))
print(lst0)
feature_importance=list(np.zeros(312))
from sklearn.ensemble import GradientBoostingClassifier
kf=KFold(n_splits=10,shuffle=True,random_state=0)
curr_score=0
spe_score=0
sen_score=0
curr_score1=0
f1_score=0
for train_index,test_index in kf.split(data):
   #  clf = svm.SVC(C=2, kernel='rbf')
   # # clt=DecisionTreeClassifier(max_depth=5,random_state=0).fit(data[train_index],target[train_index])
   #  clt3 = clf.fit(data[train_index], target[train_index])
    clt3 = GradientBoostingClassifier(max_depth=3, n_estimators=90, learning_rate=0.0525, random_state=42).fit(
        data[train_index], target[train_index])
    curr_score = curr_score + clt3.score(data[test_index], target[test_index])
    curr_score1 = curr_score1 + clt3.score(data[train_index], target[train_index])
    print('准确率为：',clt3.score(data[test_index],target[test_index]))
    print('准确率为：', clt3.score(data[train_index], target[train_index]))
    for i in range(312):
        feature_importance[i] += clt3.feature_importances_[i]
    #print(clt3.feature_importances_)
    target_exchange = []
    predict_exchange = []
    for item in target[test_index]:
        if item == 2:
            item = 0
            target_exchange.append(item)
        else:
            target_exchange.append(item)
    for item in clt3.predict_proba(list(data[test_index])):
        predict_exchange.append(item[0])
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(list(data[test_index]))):
       if list(clt3.predict(list(data[test_index])))[i] == 1:
          if list(target[test_index])[i] == 1:
              tp += 1
          else:
              fp += 1
       else:
          if list(target[test_index])[i] == 1:
              fn += 1
          else:
              tn += 1
    tnr = tn / (fp + tn+0.001)
    tpr = tp / (tp + fn+0.001)
    spe_score += tnr
    sen_score += tpr
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1score = (2 * recall * precision) / (recall + precision)
    f1_score += F1score
    print('specificity为：', tnr)
    print('sensitivity为：', tpr)
    print('F1-score为:', F1score)

avg_score3=curr_score/10
avg_spe3=spe_score/10
avg_sen3=sen_score/10
avg_f1score=f1_score/10
print('平均准确率为：',avg_score3)
print('平均specificity为：',avg_spe3)
print('平均sensitivity为:',avg_sen3)
print('平均F1score为：',avg_f1score)
mean_feature_importance=list(np.zeros(312))
for i in range(312):
    mean_feature_importance[i]=feature_importance[i]/10
copy1=copy.deepcopy(mean_feature_importance)
copy1.sort()
d=copy1[-50:]
print(d)
lst0=[]
for i in d:
    lst0.append((mean_feature_importance.index(i)))
print('-------------------------------')
print(lst0)
feature_importance=list(np.zeros(312))
import xgboost as xgb
curr_score=0
spe_score=0
sen_score=0
curr_score1=0
f1_score,a=0,0
fig=plt.figure(figsize=(8,8))
sub=fig.add_subplot(111)
sub.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0])
sub.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
ax=gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlim((0.0,1.05))
plt.ylim((0.0,1.05))
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
for train_index,test_index in kf.split(data):
    a+=1
    clt3 = xgb.XGBClassifier(max_depth=3, n_estimators=105, learning_rate=0.055, random_state=42).fit(data[train_index], target[train_index])
    curr_score = curr_score + clt3.score(data[test_index], target[test_index])
    curr_score1 = curr_score1 + clt3.score(data[train_index], target[train_index])
    print('准确率为：',clt3.score(data[test_index],target[test_index]))
    for i in range(312):
        feature_importance[i] += clt3.feature_importances_[i]
    #print(clt3.feature_importances_)
    target_exchange = []
    predict_exchange = []
    for item in target[test_index]:
        if item == 2:
            item = 0
            target_exchange.append(item)
        else:
            target_exchange.append(item)
    for item in clt3.predict_proba(list(data[test_index])):
        predict_exchange.append(item[1])
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(list(data[test_index]))):
       if list(clt3.predict(list(data[test_index])))[i] == 1:
          if list(target[test_index])[i] == 1:
              tp += 1
          else:
              fp += 1
       else:
          if list(target[test_index])[i] == 1:
              fn += 1
          else:
              tn += 1
    tnr = tn / (fp + tn+0.001)
    tpr = tp / (tp + fn+0.001)
    spe_score += tnr
    sen_score += tpr
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1score = (2 * recall * precision) / (recall + precision)
    f1_score += F1score
    print('specificity为：', tnr)
    print('sensitivity为：', tpr)
    print('F1-score为:', F1score)
    y_label = np.array(target_exchange)
    y_pred = np.array(predict_exchange)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])

    sub.plot(fpr[0], tpr[0],
             lw=2, label=r'{0}-Fold'.format(a) + ' (area = %0.4f)' % roc_auc[0])
    sub.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
avg_score3=curr_score/10
avg_spe3=spe_score/10
avg_sen3=sen_score/10
avg_f1score=f1_score/10
print('平均准确率为：',avg_score3)
print('平均specificity为：',avg_spe3)

print('平均sensitivity为:',avg_sen3)
print('平均F1score为：',avg_f1score)

mean_feature_importance=list(np.zeros(312))
for i in range(312):
    mean_feature_importance[i]=feature_importance[i]/10
#print(mean_feature_importance)
copy1=copy.deepcopy(mean_feature_importance)
copy1.sort()
d=copy1[-80:]
print(d)
lst0=[]
for i in d:
    lst0.append((mean_feature_importance.index(i)))
print(lst0)
print('-------------------------------')
plt.tight_layout()
plt.show()
