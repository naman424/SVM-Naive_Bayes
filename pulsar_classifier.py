#importing the libraries
import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
data=pd.read_csv('pulsar_stars.csv')
X=data.iloc[:,0:8].values
y=data.iloc[:,-1].values
data.columns = ['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile', 'mean_dmsnr',
               'std_dmsnr', 'kurtosis_dmsnr', 'skewness_dmsnr', 'target']
    
#Visualizing the Dataset
import seaborn as sns
plt.figure(figsize=(15,7))
vis1 = sns.countplot(data['target'], palette='OrRd')
plt.title('Distribution of target', fontsize=15)
plt.xlabel('Target', fontsize=13)
plt.ylabel('Count', fontsize=13)

for p in vis1.patches:
    vis1.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=13)



#splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import  StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#kernel SVM
from sklearn.svm import  SVC
classifier1=SVC(kernel='rbf',random_state=0)
classifier1.fit(X_train,y_train)

#naive bayes
from sklearn.naive_bayes import GaussianNB
classifier2=GaussianNB()
classifier2.fit(X_train,y_train)

#SVM
from sklearn.svm import  SVC
classifier3=SVC(kernel='linear',random_state=0)
classifier3.fit(X_train,y_train)

#predicting values for test set
y_pred_kernel_svm=classifier1.predict(X_test)
y_pred_naive_bayes=classifier2.predict(X_test)
y_pred_svm=classifier3.predict(X_test)

#confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm1=confusion_matrix(y_test,y_pred_kernel_svm)
cm2=confusion_matrix(y_test,y_pred_naive_bayes)
cm3=confusion_matrix(y_test,y_pred_svm)
as1=accuracy_score(y_test,y_pred_kernel_svm)
as2=accuracy_score(y_test,y_pred_naive_bayes)
as3=accuracy_score(y_test,y_pred_svm)

#hence accuracy of linear svm is the best followed by gaussian svm which is better than 
#naive bayes

#visualising the predicted results
xval=np.arange(len(y_test))
plt.bar(xval,y_test,color='red',label='test dataset')
plt.bar(xval,y_pred_kernel_svm,color='blue',label='gaussian SVM')
plt.bar(xval,y_pred_naive_bayes,color='green',label='naive bayes')
plt.bar(xval,y_pred_svm,color='purple',label='linear SVM')
plt.legend()
plt.show()











