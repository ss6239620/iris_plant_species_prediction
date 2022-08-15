from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


iris_dataset=load_iris()

X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

KNeighborsClassifier(algorithm='aoto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=1,p=2,weights='uniform')

#write your measurment here...
X_new=np.array([[5,2.9,3,0.2]])

prediction=knn.predict(X_new)

print('pridiction: ',format(prediction))
print('species of plant: ',format(iris_dataset['target_names'][prediction]))


y_pred=knn.predict(X_test)
print('Test pridictions: ',format(y_pred))
print('Test accuracy is: {}'.format(knn.score(X_test,y_test)))


'''For checking the validty of data using pair plotting'''
# grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',
#                         hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)


plt.show()