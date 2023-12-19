from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import sklearn.svm 
import sklearn.ensemble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def get_data (X_path, y_path, model, n):

    model_name = list(models.keys())[model]
    model = list(models.values())[model]

    # load the dataset
    X_all = pd.read_csv(X_path)
    y_all = pd.read_csv(y_path)

    # shuffle the data
    X_all, y_all = shuffle(X_all, y_all, random_state=1)

    # only use the first N samples to limit training time
    num_samples = int(len(X_all)*n)
    X, y = X_all[:num_samples], y_all[:num_samples]

    X_used = StandardScaler().fit_transform(X)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X_used, y, test_size=0.2, random_state=1)

    print("\n====", model_name, "====")
    
    model.fit(X_train, y_train)

    print("\n==== Quality of the model ====")

    y_pred = model.predict(X_test)

    #Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    #Confusion matrix
    confMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: \n\n", confMatrix, "\n")

states = ["ca","co","ne"]

models = {"SVM": sklearn.svm.SVC(C=1, kernel="rbf", probability=True),
          "Random Forest": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=1000),
          "Ada Boost": sklearn.ensemble.AdaBoostClassifier(learning_rate=1, n_estimators=1000),
          "Gradient Boosting": sklearn.ensemble.GradientBoostingClassifier(learning_rate=1, max_depth=1,  n_estimators=1000)}

for s in states : 
    
    n = 0.05 if s == "ca" else 1

    print("=======",s,"========")

    X_path = '~/Bureau/5A/app_sup/data/acsincome_' + s + '_features.csv'
    y_path = '~/Bureau/5A/app_sup/data/acsincome_' + s + '_labels.csv'

    for m, (name, model) in enumerate(models.items()):
        get_data(X_path, y_path, m, n)