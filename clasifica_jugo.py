import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
import sklearn.tree

data = pd.read_csv('OJ.csv')

data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin

data = data.drop(['Purchase'],axis=1)

predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')

x_train, x_test, y_train, y_test = train_test_split(data[predictors], data['Target'], train_size=0.5)
size_bootstraping = np.shape(x_train)[0]

f1_score_train = []
f1_score_test = []
desviaciones_train = []
desviaciones_test = []
features = []
for i in range(10):
    f1_train = []
    f1_test = []
    features_actual = []
    for j in range(100):
        seleccionados = np.random.choice(range(size_bootstraping), size_bootstraping)
        x_train_select = x_train.iloc[seleccionados,:]
        y_train_select = y_train.iloc[seleccionados]
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=i+1)
        clf.fit(x_train_select,y_train_select)
        prediccion_train = clf.predict(x_train_select)
        prediccion_test = clf.predict(x_test)
        f1_train.append(sklearn.metrics.f1_score(y_train_select, prediccion_train))
        f1_test.append(sklearn.metrics.f1_score(y_test, prediccion_test))
        features_actual.append(clf.feature_importances_)
    f1_score_train.append(np.mean(f1_train))
    f1_score_test.append(np.mean(f1_test))
    desviaciones_train.append(np.std(f1_train))
    desviaciones_test.append(np.std(f1_test))
    features.append(np.mean(features_actual, axis = 0))
    
x = np.arange(10)+1
plt.figure(figsize = (6,6))
plt.errorbar(x,f1_score_train,yerr = desviaciones_train, fmt = 'o',label = 'Train(50%)')
plt.errorbar(x,f1_score_test,yerr = desviaciones_test, fmt = 'o', label = 'Test(50%)')
plt.legend()
plt.xlabel('max depth')
plt.ylabel('Average F1-score')
plt.savefig('F1_training_test.png')

plt.figure(figsize = (6,6))
for i in range(np.shape(features)[1]):
    plt.plot(x,np.array(features)[:,i], label = 'Feature {}'.format(i+1))
    plt.legend()
plt.xlabel('max depth')
plt.ylabel('Average feature importance')
plt.savefig('features.png')