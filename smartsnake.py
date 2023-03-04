#importing relitive libs
import keras
import numpy as np
import pandas as pd

%matplotlib inline 
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import mpld3 as mpl

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

df =pd.read_csv(r"C:\Users\Asus\Desktop\ml proje\Data\data.csv"
                ,header= 0)

"""
### Algorithm of devoloped framework ###

1. Collect training sampled data consisting of 30 features.
2. Missing values are replaced by mean value for rough approximation.
3. Over-sampling of the minority class is performed using
SMOTE algorithm.
4. Rank the features using index value to measure the importance of features.
5. Split the data into bootstrap samples for independently
sampling with replacement from initial sampled data.
6. Call Adaboost Model using n bootstrap samples.
7. Combine the prediction of n models using majority voting.
8. Outcome of the prediction is winning class in the majority
voting.
"""

#Clean and prepare data
df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
# size of the dataframe
len(df)

df.diagnosis.unique()

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()

#Explore Data
df.describe()

df.describe()
plt.hist(df['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()

#nucleus features vs diagnosis¶
features_mean=list(df.columns[1:11])
# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]


###study###
#Stack the data
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, density = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()

################################################
#end of data preprocessing - Train test split

X = df.loc[:, df.columns != 'diagnosis']
y = df.iloc[:,:1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41, shuffle=(True))
################################################
###SMOTE###

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

################################################

#evaluation func

def evaluate_model(model, X_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}

from sklearn import tree

def sklearn_eval(X_train,y_train):
    dtc = tree.DecisionTreeClassifier(random_state=0)
    dtc.fit(X_train, y_train)
    
    # Evaluate Model
    dtc_eval = evaluate_model(dtc, X_test, y_test)
    
    # Print result
    print('Accuracy:', dtc_eval['acc'])
    print('Precision:', dtc_eval['prec'])
    print('Recall:', dtc_eval['rec'])
    print('F1 Score:', dtc_eval['f1'])
    print('Cohens Kappa Score:', dtc_eval['kappa'])
    print('Area Under Curve:', dtc_eval['auc'])
    print('Confusion Matrix:\n', dtc_eval['cm'])

################################################

### Randdom Forest ###

from sklearn.ensemble import RandomForestClassifier

def randomforest_eval(X_train,y_train):
    rf_clf = RandomForestClassifier(criterion='entropy')   
    rf_clf.fit(X_train,y_train)
    
    rf_eval = evaluate_model(rf_clf, X_test, y_test)
    print('Accuracy:', rf_eval['acc'])
    print('Precision:', rf_eval['prec'])
    print('Recall:', rf_eval['rec'])
    print('F1 Score:', rf_eval['f1'])
    print('Cohens Kappa Score:', rf_eval['kappa'])
    print('Area Under Curve:', rf_eval['auc'])
    print('Confusion Matrix:\n', rf_eval['cm'])

################################################

### Naive Bayes ###

from sklearn.naive_bayes import GaussianNB

def naivebayes_eval(X_train,y_train):
    #Calling the Class
    naive_bayes = GaussianNB()
     
    #Fitting the data to the classifier
    naive_bayes.fit(X_train , y_train)
     
    #Predict on test data
    y_pred = naive_bayes.predict(X_test)
    naive_eval = evaluate_model(naive_bayes, X_test, y_test)
    print('Accuracy:', naive_eval['acc'])
    print('Precision:', naive_eval['prec'])
    print('Recall:', naive_eval['rec'])
    print('F1 Score:', naive_eval['f1'])
    print('Cohens Kappa Score:', naive_eval['kappa'])
    print('Area Under Curve:', naive_eval['auc'])
    print('Confusion Matrix:\n', naive_eval['cm'])

################################################

### Ada Boost ###

from sklearn.ensemble import AdaBoostClassifier

def adaboost(X_train,y_train):
    abc = AdaBoostClassifier(n_estimators=50,
             learning_rate=1)
    ada_boost = abc.fit(X_train, y_train)
    y_pred = ada_boost.predict(X_test)
    ada_beval = evaluate_model(ada_boost, X_test, y_test)
    print('Accuracy:', ada_beval['acc'])
    print('Precision:', ada_beval['prec'])
    print('Recall:', ada_beval['rec'])
    print('F1 Score:', ada_beval['f1'])
    print('Cohens Kappa Score:', ada_beval['kappa'])
    print('Area Under Curve:', ada_beval['auc'])
    print('Confusion Matrix:\n', ada_beval['cm'])
    

#cols = [20,0,26,25,21,1,2,22,6,27]
#z = X.iloc[:,cols].copy()

#top 5 feature
#cols = [0,1,3,8,9]
#z1 = z.iloc[:,cols].copy()

# acc of models
sklearn_eval(X_train,y_train)
randomforest_eval(X_train,y_train)
naivebayes_eval(X_train,y_train)
adaboost(X_train,y_train)


#execute the funtions from ml-paper.py
sklearn_eval(X_res,y_res)
randomforest_eval(X_res,y_res)
naivebayes_eval(X_res,y_res)
adaboost(X_res,y_res)


################################################

## function to generate bootstrap datasets ##
def make_bootstraps(data,n_bootstraps=3):
    #initialize output dictionary & unique value count
    dc   = {}
    unip = 0
    #get sample size
    b_size = data.shape[0]
    #get list of row indexes
    idx = [i for i in range(b_size)]
    #loop through the required number of bootstraps
    for b in range(n_bootstraps):
        #obtain boostrap samples with replacement
        sidx   = np.random.choice(idx,replace=True,size=b_size)
        b_samp = data[sidx,:]
        #compute number of unique values contained in the bootstrap sample
        unip  += len(set(sidx))
        #obtain out-of-bag samples for the current b
        oidx   = list(set(idx) - set(sidx))
        o_samp = np.array([])
        if oidx:
            o_samp = data[oidx,:]
        #store results
        dc['boot_'+str(b)] = {'boot':b_samp,'test':o_samp}
    #state the mean number of unique values in the bootstraps
    print('Mean number of unique values in each bootstrap: {:.2f}'.format(unip/n_bootstraps))
    #return the bootstrap results
    return(dc)

#perform bootstrap
data = X_train.to_numpy()
dcBoot = make_bootstraps(data)


### we need to return to dataframes to continue
df1 = pd.DataFrame.from_dict(dcBoot)
df2 = df1.iloc[:,1].copy()


































