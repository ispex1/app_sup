from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import sklearn.svm 
import sklearn.ensemble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import seaborn as sns

def corr_heatmap(init_corr,final_corr,model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(init_corr.iloc[:-1,:-1], annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix beetwen features and label - %s' % model_name)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(final_corr.iloc[:-1,:-1], annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix beetwen features and label predicted - %s' % model_name)
    plt.show()

def corr_barplot(init_corr,final_corr,model_name):
    plt.figure(figsize=(6, 6))
    pal = sns.color_palette("coolwarm", len(init_corr))
    rank = init_corr['label'].argsort().argsort() 

    sns.barplot(x=final_corr.index, y=final_corr['label'], palette=np.array(pal[::-1])[rank])
    plt.title('Correlation Matrix beetwen features and label predicted - %s' % model_name)

    plt.show()

def get_data(model,n):

    model_name = list(models.keys())[model]
    model = list(models.values())[model]

    # load the dataset
    X_all = pd.read_csv('~/Bureau/5A/app_sup/data/acsincome_ca_features.csv')
    y_all = pd.read_csv('~/Bureau/5A/app_sup/data/acsincome_ca_labels.csv')

    # shuffle the data
    X_all, y_all = shuffle(X_all, y_all, random_state=1)

    # only use the first N (1%) samples to limit training time
    num_samples = int(len(X_all)*n)
    X, y = X_all[:num_samples], y_all[:num_samples]

    X_used = StandardScaler().fit_transform(X)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X_used, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test

models = {"SVM": sklearn.svm.SVC(C=1, kernel="rbf", probability=True),
          "Random Forest": sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=1000),
          "Ada Boost": sklearn.ensemble.AdaBoostClassifier(learning_rate=1, n_estimators=1000),
          "Gradient Boosting": sklearn.ensemble.GradientBoostingClassifier(learning_rate=1, max_depth=1,  n_estimators=1000)}

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
fig.suptitle('Importance de chaque feature', fontsize=16)

for m, (name, model) in enumerate(models.items()):
    X_train, X_test, y_train, y_test = get_data(m, 0.05)

    print("\n========", name, "========")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    """# Define dataframe with the features
    df_train = pd.DataFrame(X_train)
    df_train['label'] = y_train['PINCP'].tolist()
    df_test = pd.DataFrame(X_test)
    df_test['label'] = y_pred

    #Correlation between features and labels
    init_corr = df_train.corr(method='pearson')
    final_corr = df_test.corr(method='pearson')

    
    corr_heatmap(init_corr,final_corr,model_name)
    corr_barplot(init_corr.iloc[:-1],final_corr.iloc[:-1],model_name)
    
    """

    df_train2 = pd.DataFrame(X_train)
 
    #Permutation importance
    res = permutation_importance(model, X_train, y_train)

    
    print(f"{name} : {res['importances_mean']}")
    
    sns.barplot(x=res['importances_mean'], y=X_train.columns, ax=axes[m//2, m%2])
    axes[m//2, m%2].set_title(name)

plt.tight_layout()
plt.show()

