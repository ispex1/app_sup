from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
import sklearn.svm 
import sklearn.ensemble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#show the distribution of the different races
def distribution(path) :
    
    etud = np.array(pd.read_csv(path)).reshape(-1)
    name_possible_race = ["White", "Black", "American Indian", "Alaska", "American Indian or Alaska", "Asian", "Pacific Islander", "Other", "More than 1 race"]
    repartition = dict(sorted(dict(Counter(list(etud))).items()))

    print(repartition)

    plt.bar(name_possible_race, repartition.values())
    plt.xticks(rotation=90)
    plt.ylabel("amount")
    plt.show()

def equities (X_path, y_path, etud_path, model, n, race):

    model_name = list(models.keys())[model]
    model = list(models.values())[model]

    # load the dataset
    X_all = pd.read_csv(X_path)
    y_all = pd.read_csv(y_path)

    #etud can be sex or race
    etud_all = np.array(pd.read_csv(etud_path))

    # shuffle the data
    X_all, y_all, etud_all = shuffle(X_all, y_all, etud_all, random_state=1)

    # only use the first N samples to limit training time
    num_samples = int(len(X_all)*n)
    X, y, etud = X_all[:num_samples], y_all[:num_samples], etud_all[:num_samples]

    X_used = StandardScaler().fit_transform(X)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test, etud_train, etud_test = \
        train_test_split(X_used, y, etud, test_size=0.2, random_state=1)

    print("\n#############################", model_name, "#############################")
    
    model.fit(X_train, y_train)

    print("X train:", X_train[:,8])

    # Split between etud features
    X_g1_train = [X_train[i] for i in range(X_train.shape[0]) if etud_train[i][0] == 1]
    y_g1_train = [y_train['PINCP'].iat[i] for i in range(len(y_train)) if etud_train[i][0] == 1]
    X_g2_train = [X_train[i] for i in range(X_train.shape[0]) if etud_train[i][0] != 1]
    y_g2_train = [y_train['PINCP'].iat[i] for i in range(len(y_train)) if etud_train[i][0] != 1]
    X_g1_test = [X_test[i] for i in range(X_test.shape[0]) if etud_test[i][0] == 1]
    y_g1_test = [y_test['PINCP'].iat[i] for i in range(len(y_test)) if etud_test[i][0] == 1]
    X_g2_test = [X_test[i] for i in range(X_test.shape[0]) if etud_test[i][0] != 1]
    y_g2_test = [y_test['PINCP'].iat[i] for i in range(len(y_test)) if etud_test[i][0] != 1]
    


    print("\n==== Quality of the model ====")
    
    y_g1_pred = model.predict(X_g1_test)
    y_g2_pred = model.predict(X_g2_test)

    #Accuracy
    accuracy_g1 = accuracy_score(y_g1_test, y_g1_pred)
    print("Accuracy for g1 (", "whites" if race else "male",") prediction : ", accuracy_g1)

    accuracy_g2 = accuracy_score(y_g2_test, y_g2_pred)
    print("Accuracy for g2 (", "other races" if race else "female",") prediction : ", accuracy_g2)

    #Confusion matrix
    confMatrix_g1 = confusion_matrix(y_g1_test, y_g1_pred)
    print("Confusion matrix for male: \n\n", confMatrix_g1, "\n")

    confMatrix_g2 = confusion_matrix(y_g2_test, y_g2_pred)
    print("Confusion matrix for g2 (", "other races" if race else "female",") : \n\n", confMatrix_g2, "\n")

    #statistical parity
    print("\n==== Statistical parity ====")
    print("comparaisons des taux de prédictions positives")

    parity_g1 = np.sum(confMatrix_g1[:,0])/np.sum(confMatrix_g1)
    print("statistical parity for male confusion matrix :", parity_g1)

    parity_g2 = np.sum(confMatrix_g2[:,0])/np.sum(confMatrix_g2)
    print("statistical parity for g2 (", "other races" if race else "female",") confusion matrix :", parity_g2)


    #equal opportunity
    print("\n==== Equal opportunity ====")
    print("comparer les taux de vrais positifs")

    opportunity_g1 = confMatrix_g1[0,0] / np.sum(confMatrix_g1[0,:])
    print("equal opportunity for male : ", opportunity_g1)

    opportunity_g2 = confMatrix_g2[0,0] / np.sum(confMatrix_g2[0,:])
    print("equal opportunity for g2 (", "other races" if race else "female",") : ", opportunity_g2)

    #predictive equality
    print("\n==== Predictive equality ====")
    print("comparer les taux de faux positifs")

    pe_g1= confMatrix_g1[1,0] / np.sum(confMatrix_g1[1,:])
    print("predictive equality for male : ", pe_g1)
    
    pe_g2= confMatrix_g2[1,0] / np.sum(confMatrix_g2[1,:])
    print("predictive equality for g2 (", "other races" if race else "female",") : ", pe_g2)


models = {"SVM": sklearn.svm.SVC(C=1000, kernel="rbf", probability=True),
        "Random Forest": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=1000),
        "Gradient Boosting": sklearn.ensemble.GradientBoostingClassifier(learning_rate=1, max_depth=10,  n_estimators=100),
        "Ada Boost": sklearn.ensemble.AdaBoostClassifier(learning_rate=1, n_estimators=1000)}

#Path
X_path_without_sex = './data/acsincome_ca_features_without_sex.csv'
X_path_without_race = './data/acsincome_ca_features_without_race.csv'
X_path = './data/acsincome_ca_features.csv'
y_path = './data/acsincome_ca_labels.csv'
sex_path = './data/acsincome_ca_group_Sex.csv'
race_path = './data/acsincome_ca_group_Race.csv'

distribution(race_path)

"""for m, (name, model) in enumerate(models.items()):
    equities(X_path, y_path, sex_path, m, 0.05, True)"""