from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.svm 
import sklearn.ensemble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# load the dataset
X_all = pd.read_csv('~/Bureau/5A/app-sup/data/acsincome_ca_features.csv')
y_all = pd.read_csv('~/Bureau/5A/app-sup/data/acsincome_ca_labels.csv')

# shuffle the data
X_all, y_all = shuffle(X_all, y_all, random_state=1)

# only use the first N samples to limit training time
num_samples = int(len(X_all)*0.01)
X, y = X_all[:num_samples], y_all[:num_samples]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1)

# print the results
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

#print selected datas
print("==== Features train ====")
print(X_train.head(5))
print("==== Label train ====")
print(y_train.head(5))

"""Les méthodes d’apprentissage étudiées dans ces TP sont les suivantes : SVM, RandomForest, AdaBoost
et GradientBoosting. Il est important d’avoir compris le principe de ces méthodes. Pour chaque
méthode, vous devez :
(1) Mettre en place une validation croisée
(2) Evaluer la qualité d’un modèle d’apprentissage en utilisant différentes métriques (accuracy,
classification_report, confusion_matrix)
(3) Mettre en place une recherche des bons hyperparamètres (gridsearchCV"""


def modelisation(model_name):
    
    # (1) Mettre en place une validation croisée
    print("\n====", model_name, "====")
    
    match model_name:
        case "SVM":
            model = sklearn.svm.SVC(probability=True)
        case "RandomForest":
            model = sklearn.ensemble.RandomForestClassifier()
        case "AdaBoost":
            model = sklearn.ensemble.AdaBoostClassifier()
        case "GradientBoosting":
            model = sklearn.ensemble.GradientBoostingClassifier()
        case _:
            print("Model not found")
            return

    crossScore = cross_val_score(model, X_train, y_train, cv=5)

    print("\n==== Results Cross Validation ====")
    print("Cross validation score: ", crossScore)
    print("Cross validation score mean: ", crossScore.mean())

    # (2) Evaluer la qualité d’un modèle d’apprentissage en utilisant différentes métriques (accuracy, classification_report, confusion_matrix)

    print("\n==== Quality of the model ====")
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time() - t0
    print("Training time (model.fit) : ", t1, "s")

    y_pred = model.predict(X_test)

    #Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    #Classification report
    classReport = classification_report(y_test, y_pred)
    print("Classification report: \n\n", classReport, "\n")

    #Confusion matrix
    confMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: \n\n", confMatrix, "\n")

    # (3) Mettre en place une recherche des bons hyperparamètres (gridsearchCV)

    print("\n==== GridSearchCV ====")
    switcher = {
        "SVM": {'C':[1, 10, 100, 1000], 'kernel': [ 'rbf', 'poly']},
        "RandomForest": {'n_estimators': [10, 100, 1000], 'max_depth': [None, 10, 100]},
        "AdaBoost": {'n_estimators': [10, 100, 1000], 'learning_rate': [0.1, 1, 10]},
        "GradientBoosting": {'n_estimators': [10, 100, 1000], 'learning_rate': [0.1, 1, 10], 'max_depth': [None, 10, 100]}
    }
    clf = GridSearchCV(model, switcher.get(model_name))
    clf.fit(X_train, y_train)
    print("Best parameters: ", clf.best_params_)
    print("Best score: ", clf.best_score_)
    print("Best estimator: ", clf.best_estimator_)
    print("Best index: ", clf.best_index_)
    print("Scorer: ", clf.scorer_)
    print("Refit time: ", clf.refit_time_)
    cv_result = clf.cv_results_
    #print("Predict log proba: ", clf.predict_log_proba(X_test))
    #print("Predict proba: ", clf.predict_proba(X_test))
    print("Score: ", clf.score(X_test, y_test))
    return y_pred

'''
# Models call
modelisation("SVM")
modelisation("RandomForest")
modelisation("AdaBoost")
modelisation("GradientBoosting")
'''

# The best model is AdaBoostClassifier with the best score of 0.7953837961825181
# The best parameters are {'learning_rate': 0.1, 'n_estimators': 1000}

# Define the optimal model
def modelisationAdaBoost():
    model = sklearn.ensemble.AdaBoostClassifier(learning_rate=0.1, n_estimators=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
y_pred = modelisationAdaBoost()

# Define a dataframe with the features
df_train = pd.DataFrame(X_train)

# Add label init to the dataframe : y_train
df_train['label'] = y_train 
init_corr = df_train.corr()
print("Corrélations initiale entre chaque feature et le label :\n", init_corr.iloc[:-1, -1])

# Delete label init
del df_train['label']
df_train['label'] = y_pred
final_corr = df_train.corr()
print("Corrélations finale entre chaque feature et le label :\n", final_corr.iloc[:-1, -1])







