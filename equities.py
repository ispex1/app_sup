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

def equities (X_path, y_path, sex_path, model, n):

    model_name = list(models.keys())[model]
    model = list(models.values())[model]

    sex_all = np.array(pd.read_csv(sex_path))

    # load the dataset
    X_all = pd.read_csv(X_path)
    y_all = pd.read_csv(y_path)

    # shuffle the data
    X_all, y_all, sex_all = shuffle(X_all, y_all,sex_all, random_state=1)

    # only use the first N samples to limit training time
    num_samples = int(len(X_all)*n)
    X, y, sex = X_all[:num_samples], y_all[:num_samples], sex_all[:num_samples]

    X_used = StandardScaler().fit_transform(X)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test, sex_train, sex_test = \
        train_test_split(X_used, y, sex, test_size=0.2, random_state=1)

    print("\n#############################", model_name, "#############################")
    
    model.fit(X_train, y_train)

    # Split between sex features
    X_male_train = [X_train[i] for i in range(X_train.shape[0]) if sex_train[i][0] == 1]
    y_male_train = [y_train['PINCP'].iat[i] for i in range(len(y_train)) if sex_train[i][0] == 1]
    X_female_train = [X_train[i] for i in range(X_train.shape[0]) if sex_train[i][0] == 2]
    y_female_train = [y_train['PINCP'].iat[i] for i in range(len(y_train)) if sex_train[i][0] == 2]
    X_male_test = [X_test[i] for i in range(X_test.shape[0]) if sex_test[i][0] == 1]
    y_male_test = [y_test['PINCP'].iat[i] for i in range(len(y_test)) if sex_test[i][0] == 1]
    X_female_test = [X_test[i] for i in range(X_test.shape[0]) if sex_test[i][0] == 2]
    y_female_test = [y_test['PINCP'].iat[i] for i in range(len(y_test)) if sex_test[i][0]== 2]



    print("\n==== Quality of the model ====")

    y_male_pred = model.predict(X_male_test)
    y_female_pred = model.predict(X_female_test)

    #Accuracy
    accuracy_male = accuracy_score(y_male_test, y_male_pred)
    print("Accuracy for male prediction : ", accuracy_male)

    accuracy_female = accuracy_score(y_female_test, y_female_pred)
    print("Accuracy for female prediction : ", accuracy_female)

    #Confusion matrix
    confMatrix_male = confusion_matrix(y_male_test, y_male_pred)
    print("Confusion matrix for male: \n\n", confMatrix_male, "\n")

    confMatrix_female = confusion_matrix(y_female_test, y_female_pred)
    print("Confusion matrix for female: \n\n", confMatrix_female, "\n")

    #statistical parity
    print("\n==== Statistical parity ====")
    print("comparaisons des taux de prédictions positives")

    parity_male = np.sum(confMatrix_male[:,0])/np.sum(confMatrix_male)
    print("statistical parity for male confusion matrix :", parity_male)

    parity_female = np.sum(confMatrix_female[:,0])/np.sum(confMatrix_female)
    print("statistical parity for female confusion matrix :", parity_female)

    #equal opportunity
    print("\n==== Equal opportunity ====")
    print("comparer les taux de vrais positifs")

    opportunity_male = confMatrix_male[0,0] / np.sum(confMatrix_male[0,:])
    print("equal opportunity for male : ", opportunity_male)

    opportunity_female = confMatrix_female[0,0] / np.sum(confMatrix_female[0,:])
    print("equal opportunity for female : ", opportunity_female)

    #predictive equality
    print("\n==== Predictive equality ====")
    print("comparer les taux de faux positifs")

    pe_male= confMatrix_male[1,0] / np.sum(confMatrix_male[1,:])
    print("predictive equality for male : ", pe_male)
    
    pe_female= confMatrix_female[1,0] / np.sum(confMatrix_female[1,:])
    print("predictive equality for female : ", pe_female)

models = {"SVM": sklearn.svm.SVC(C=1000, kernel="rbf", probability=True),
        "Random Forest": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=1000),
        "Gradient Boosting": sklearn.ensemble.GradientBoostingClassifier(learning_rate=1, max_depth=10,  n_estimators=100),
        "Ada Boost": sklearn.ensemble.AdaBoostClassifier(learning_rate=1, n_estimators=1000)}

#pour tester avec le jeu de donnée sans la feature sex :
#X_path = './data/acsincome_ca_features_without_sex.csv'
X_path = './data/acsincome_ca_features.csv'
y_path = './data/acsincome_ca_labels.csv'
sex_path = './data/acsincome_ca_group_Sex.csv'

for m, (name, model) in enumerate(models.items()):
    equities(X_path, y_path, sex_path, m, 0.05)