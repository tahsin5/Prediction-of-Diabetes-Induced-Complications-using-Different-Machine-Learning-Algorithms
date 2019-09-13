import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../Dataset/missForest_imputed_heart_v2.csv', encoding='latin-1')
df = df.dropna(subset=['Cardiovascular Risk'])
df = df.replace(r'\s+', np.nan, regex=True)
#df = df.loc[df['History.of.kidney.disease'] == 0]

x = df.iloc[:,0:-1].values                         
y = df.iloc[:,-1].values

#Convert to Categorical Variables
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1,5,6,7,8,9,10,11,12,13,15,16,17,18,19,149,150,151,152,153,154,155,156])
x = onehotencoder.fit_transform(x).toarray()
x = np.delete(x, [1,2,5,7,9,11,13,15,17,19,21,23,25,28,30,32,37,42,47,52,58,64,70], 1) 

unique_elements, counts_elements = np.unique(y, return_counts=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0,stratify=y)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=0,ratio='minority')
# X_train, y_train = sm.fit_sample(X_train, y_train)
# unique_elements, counts_elements = np.unique(y_test, return_counts=True)

colors = ["magenta","green","blue","sienna"]
recall = []
precision = [] 
F1_Score = []
AP = []
AUC = []
C = list(np.arange(0,2,0.01))
C[0] = 0.01
#C = [0.01,0.1,1,100,1000]

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
for i in range(0,200,1):
   
  classifier = LogisticRegression(random_state=0, C=C[i])
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  y_pred_prob = classifier.predict_proba(X_test)[:, 1]
   
  recall.append(metrics.recall_score(y_test, y_pred))
  precision.append(metrics.precision_score(y_test, y_pred))
  F1_Score.append(metrics.f1_score(y_test, y_pred))
  AP.append(metrics.average_precision_score(y_test, y_pred_prob))
  AUC.append(metrics.roc_auc_score(y_test, y_pred_prob))
   
max_recall = max(recall)
max_recall_index = [i for i, j in enumerate(recall) if j == max_recall]
max_precision = max(precision)
max_precision_index = [i for i, j in enumerate(precision) if j == max_precision]
max_F1Score = max(F1_Score)
max_F1Score_index = [i for i, j in enumerate(F1_Score) if j == max_F1Score]
max_AP = max(AP)
max_AP_index = [i for i, j in enumerate(AP) if j == max_AP]

fig, ax = plt.subplots()
fig.set_size_inches(9, 9)
ax.plot(C, recall, lw=2.5, label="Recall",color=colors[0])
ax.plot(C, precision, lw=2.5, label="Precision",color=colors[1])
ax.plot(C, F1_Score, lw=2.5, label="F1_Score",color=colors[2])
ax.plot(C, AP, lw=2.5, label="Average Precision",color=colors[3])
#ax.plot([94, 94], [0.6, 0.96], color='black', lw=1, linestyle='--')

plt.xlabel('Inverse of Regularization Strength(C)',fontsize=17)
plt.ylabel('Values',fontsize=17)
plt.tick_params(labelsize=14)
plt.title('Cardiovascular_Scores_Logistic_Regression',fontsize=20)
plt.legend(loc=9, bbox_to_anchor=(0.5, 0.1), ncol=4, prop={'size': 15})

plt.show()
fig.savefig('../Cardiovascular_Scores_Logistic_Regression.png', dpi=100)    
