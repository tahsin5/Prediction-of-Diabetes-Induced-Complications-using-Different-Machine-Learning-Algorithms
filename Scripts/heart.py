import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../Dataset/missForest_imputed_heart_v2.csv', encoding='latin-1')
df = df.dropna(subset=['Cardiovascular Risk'])
df = df.replace(r'\s+', np.nan, regex=True)

x = df.iloc[:,0:-1].values                         
y = df.iloc[:,-1].values
unique_elements, counts_elements = np.unique(y, return_counts=True)

#Convert to Categorical Variables
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1,5,6,7,8,9,10,11,12,13,15,16,17,18,19,149,150,151,152,153,154,155,156])
x = onehotencoder.fit_transform(x).toarray()
x = np.delete(x, [1,2,5,7,9,11,13,15,17,19,21,23,25,28,30,32,37,42,47,52,58,64,70], 1) 

unique_elements, counts_elements = np.unique(y, return_counts=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

colors = ["red","green","magenta","black","yellow","cyan"]
fig = plt.figure()
fig.set_size_inches(9, 9)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate or (1-Specificity)',fontsize=17)
plt.ylabel('True Positive Rate or Sensitivity',fontsize=17)
plt.tick_params(labelsize=14)
plt.title('Cardiovascular ROC Curves',fontsize=20)
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess(AUC = 0.5)')

LR = LogisticRegression(random_state=0)
Ada = AdaBoostClassifier(DecisionTreeClassifier(random_state=0),
                   algorithm="SAMME", n_estimators=200,random_state=0)
DT = DecisionTreeClassifier(random_state=0)
RF = RandomForestClassifier(n_estimators=80,random_state=0)
svm = SVC(random_state=0, probability=True)
NB = GaussianNB()

algos = [LR, NB, DT, Ada, RF, svm]
algo_names = ['LR', 'NB', 'DT', 'Ada', 'RF','SVM']
accuracy = []
precisions = []
recalls = []
F1_score = []
AP = []
brier = []
AUC = []

from sklearn import metrics
for i in range(0,len(algos)-1):
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0,stratify=y)
    clf = algos[i]
    
    if algo_names[i] == 'LR':
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc = sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    recalls.append(metrics.recall_score(y_test, y_pred))
    precisions.append(metrics.precision_score(y_test, y_pred))
    F1_score.append(metrics.f1_score(y_test, y_pred))
    AP.append(metrics.average_precision_score(y_test, y_pred_prob)) 
    brier.append(metrics.brier_score_loss(y_test, y_pred))
    AUC.append(metrics.roc_auc_score(y_test, y_pred_prob))
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, color=colors[i], lw=1.5, label=algo_names[i]+'(AUC = %0.2f)' % AUC[i])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0,stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc = sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)    
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
y_pred_prob = svm.predict_proba(X_test)[:,1]

accuracy.append(metrics.accuracy_score(y_test, y_pred))
recalls.append(metrics.recall_score(y_test, y_pred))
precisions.append(metrics.precision_score(y_test, y_pred))
AP.append(metrics.average_precision_score(y_test, y_pred_prob)) 
brier.append(metrics.brier_score_loss(y_test, y_pred))
AUC.append(metrics.roc_auc_score(y_test, y_pred_prob))
F1_score.append(metrics.f1_score(y_test, y_pred))
    
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, color=colors[5], lw=1.5, label='SVM(AUC = %0.2f)' % AUC[5])

plt.legend(loc="lower right", prop={'size': 12})
plt.show()
fig.savefig('../../Docs/images/Results/Cardiovascular_ROC_Updated.png', dpi=100)

#Precision-Recall Curve
# from sklearn.metrics import precision_recall_curve
# from sklearn.utils.fixes import signature

# fig = plt.figure()
# fig.set_size_inches(9,9)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall',fontsize=15)
# plt.ylabel('Precision',fontsize=15)
# plt.title('Cardiovascular PR Curves',fontsize=18)

# for i in range(0,len(algos)-1):
    
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0,stratify=y)
#     clf = algos[i]
    
#     if algo_names[i] == 'LR':
#         from sklearn.preprocessing import StandardScaler
#         sc = StandardScaler()
#         sc = sc.fit(X_train)
#         X_train = sc.transform(X_train)
#         X_test = sc.transform(X_test)
    
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
#     #precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
#     #step_kwargs = ({'step': 'post'}
#     #           if 'step' in signature(plt.fill_between).parameters
#     #           else {})
#     #plt.step(recall, precision, color=colors[i], lw=1.5, alpha=0.5,
#     #     where='post',label=algo_names[i]+'(AP = %0.2f)' % AP[i])

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc = sc.fit(X_train)
# X_train = sc.transform(X_train)
# X_test = sc.transform(X_test)    
# svm.fit(X_train, y_train)
# y_pred = svm.predict(X_test)
# y_pred_prob = svm.predict_proba(X_test)[:, 1]

# precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})
# plt.step(recall, precision, color='black', lw=3, alpha=0.5,
#          where='post',label=algo_names[5]+'(AP = %0.2f)' % AP[5])

# plt.legend(loc="upper right")
# plt.show()
# fig.savefig('../../Docs/images/Results/PRCurve_Cardiovascular_SVM.png', dpi=100)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
    

