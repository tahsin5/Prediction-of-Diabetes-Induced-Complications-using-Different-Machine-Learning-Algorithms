import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../Dataset/missForest_imputed.csv', encoding='latin-1')
df = df.replace(r'\s+', np.nan, regex=True)
df = df.replace(r'?', np.nan)
df = df.loc[df['History.of.kidney.disease'] == 0]

x = df.iloc[:,0:-1].values                         
y = df.iloc[:,-1].values

feature_names = df.columns.values

#item = np.where(x=='3.1???')
#y = df.loc[:,'History of kidney disease'].values

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Convert to Categorical Variables
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,149,150,151,152,153,154,155,156])
X = onehotencoder.fit_transform(x).toarray()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(0.8)                         #80% of variance retained
pca.fit(X_scaled)
X_scaled_pca = pca.transform(X_scaled)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_pca, y, test_size = 0.3, random_state = 0)


explained_variance = pca.explained_variance_ratio_
cumulative_variance=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
components = pca.components_
singularValue = pca.singular_values_
numberofcomponents = pca.n_components_


##PCA  Variance plot
#plt.plot(cumulative_variance)
#plt.xlabel("Principal Components")
#plt.ylabel("Cumulative Explained Variance")
#plt.plot([0, 50], [80, 80], color='k', linestyle='--', linewidth=1.5)
#plt.plot([50, 50], [80, 0], color='k', linestyle='--', linewidth=1.5)
#plt.savefig("pca_variance.png")


##PCA Data Plot
#x_values = np.arange(0.0, 778.0)
#from itertools import cycle
#cycol = cycle('bgrcmk')
#fig = plt.figure()
#fig.set_size_inches(12, 9)
#
#for y in range(50):
#    plt.scatter(x_values,X_scaled_pca[:,y], c=next(cycol),alpha=.6)   
#
#plt.ylabel("All Variables")
#plt.xlabel("Patient Index")
#plt.grid(True)
#
#plt.show()
#fig.savefig('pca.png', dpi=100)


##plot PCA first and second principal components
#plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
#plt.xlabel("first principal component")
#plt.ylabel("second principal component")


##plot all principal components
#plt.imshow(components.T)
#plt.yticks(range(len(feature_names)), feature_names)
#plt.colorbar()


##plot variables relating to each component
#plt.figure(figsize=(11, 11))
#plt.scatter(components[0], components[1])
#for i, feature_contribution in enumerate(components.T):
#    plt.annotate(feature_names[i], feature_contribution)
#plt.xlabel("first principal component")
#plt.ylabel("second principal component")


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test,y_pred)
print("Kidney Prediction\n")
print("Accuracy:",ac,"\n")
print("Confusion Matrix:")
print(cm,"\n")
print("Classification Report:\n")
print(cr)

#plot cm
##import matplotlib.pyplot as plt
##import numpy as np
##import itertools
##
##cmap = None
##normalize = False
##target_names = ['0', '1']
##title = "Confusion Matrix"
##
##accuracy = np.trace(cm) / float(np.sum(cm))
##misclass = 1 - accuracy
##
##if cmap is None:
##    cmap = plt.get_cmap('Blues')
##
##plt.figure(figsize=(8, 6))
##plt.imshow(cm, interpolation='nearest', cmap=cmap)
##plt.title(title)
##plt.colorbar()
##
##if target_names is not None:
##        tick_marks = np.arange(len(target_names))
##        plt.xticks(tick_marks, target_names, rotation=45)
##        plt.yticks(tick_marks, target_names)
##
##if normalize:
##        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
##
##
##thresh = cm.max() / 1.5 if normalize else cm.max() / 2
##for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
##    if normalize:
##            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
##                     horizontalalignment="center",
##                     color="white" if cm[i, j] > thresh else "black")
##    else:
##            plt.text(j, i, "{:,}".format(cm[i, j]),
##                     horizontalalignment="center",
##                     color="white" if cm[i, j] > thresh else "black")
##            
##    plt.tight_layout()
##    plt.ylabel('True label')
##    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
##    plt.show()

