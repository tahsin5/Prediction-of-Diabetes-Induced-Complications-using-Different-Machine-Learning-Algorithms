import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#df = pd.read_csv('../../Dataset/Kidney_mean_imputed.csv', encoding='latin-1')
#df2 = pd.read_csv('../../Dataset/Generic_mean_imputed.csv', encoding='latin-1')
#df = pd.concat([df2, df], axis=1)

df = pd.read_csv('../Dataset/missForest_imputed_heart_v2.csv', encoding='latin-1')
df = df.dropna(subset=['Cardiovascular Risk'])
df = df.replace(r'\s+', np.nan, regex=True)
#df = df.loc[df['History.of.kidney.disease'] == 0]

x = df.iloc[:,0:-1].values                         
y = df.iloc[:,-1].values
unique_elements, counts_elements = np.unique(y, return_counts=True)

#Convert to Categorical Variables
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,149,150,151,152,153,154,155,156])
x = onehotencoder.fit_transform(x).toarray()

unique_elements, counts_elements = np.unique(y, return_counts=True)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(0.8)                         #80% of variance retained
#pca.fit(x)
#x = pca.transform(x)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


#clf = LogisticRegression(random_state=0)
#clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=0),
#                   algorithm="SAMME", n_estimators=200,random_state=0)
#clf = DecisionTreeClassifier(random_state=0)

#clf = SVC(random_state=0)
#clf = GaussianNB()
clf = RandomForestClassifier(n_estimators=80,random_state=0)

sm = SMOTE(random_state=0,ratio='minority')


pipeline = make_pipeline(sm, clf)

from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(pipeline,x,y,cv=10)

#Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 
cm = confusion_matrix(y, y_pred)
ac = accuracy_score(y, y_pred)
cr = classification_report(y,y_pred)

print("Cardiovascular Risk Prediction\n")
print("Accuracy:",ac,"\n")
print("Confusion Matrix:")
print(cm,"\n")
print("Classification Report:\n")
print(cr)

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)
specificity = 1 - fpr[1]
print("False Positive Rate:",fpr[1])
print("Specificity:",specificity)
print("AUC:",roc_auc)

fig = plt.figure()
fig.set_size_inches(6, 6)
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate or (1-Specificity)')
plt.ylabel('True Positive Rate or Sensitivity')
plt.title('ROC Curve-RF Oversampled')
plt.legend(loc="lower right")
plt.show()
fig.savefig('../../Docs/images/Results/Cardiovascular/ROC_RF_OS.png', dpi=100)

#plot cm
import itertools

cmap = plt.cm.Blues
classes = ['0', '1']
title = "Confusion Matrix: RF Oversampled"

print('Confusion matrix Plot')

fig = plt.figure()
fig.set_size_inches(5, 5)
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
    horizontalalignment="center",
    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.show()
fig.savefig('../../Docs/images/Results/Cardiovascular/ConfusionMatrix_RF_OS.png', dpi=100)
