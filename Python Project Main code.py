# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:35:33 2018

@author: Parth Girme, Eugene Olkhov, Matthew Shaw
"""

                    ######################################
                    ##        PYTHON PROJECT            ##
                    ######################################


              ############ Online Reviews for Mashable ##########
                    

## importing relevant packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
##NOTE: The code includes seaborn scatterplots, which require an updated version of seaborn. May not run.
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestRegressor

##-------------------------------------------------------------------------------------------------------

## downloading the data & removing variables ##
df = pd.read_csv('OnlineNewsPopularity.csv')
df_main = pd.read_csv('OnlineNewsPopularity.csv')
df.drop(['url', ' timedelta', ' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max',' kw_max_max',
         ' kw_avg_max',' kw_min_avg',' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares',
         ' self_reference_max_shares', ' self_reference_avg_sharess', ' is_weekend',
         ' global_subjectivity', ' title_subjectivity', ' abs_title_subjectivity'],
         axis = 1, inplace = True)
##-- We downloaded data from the UCI depository, unfortuantely some of the variable descriptions were 
##-- not clear and hence we had to remove said variables from the beginning. Also created a main data file
##-------------------------------------------------------------------------------------------------------

## converting categorical variables to 'object' type & running a correlation
threshold = 7
nan = []
for i in df:
    if df[i].nunique() <= threshold:
        nan.append(i)
        df[nan] = df[nan].astype(object)

a = df.corr()
##-- We converted the categorical variables to object by setting a threshold on the number of unique
##-- obsrvations. In this case we set the threshold to 7(we knew the leat unique number of observation
##-- for continuous variables) and said all the variable with unique number of observations under 7 are
##-- categorical, which they are. 
##-------------------------------------------------------------------------------------------------------

## Further removing the variables based on correlation
df.drop([' n_non_stop_words', ' n_non_stop_unique_tokens', ' global_rate_positive_words'
             , ' global_rate_negative_words', ' rate_positive_words', ' rate_negative_words'
             , ' min_positive_polarity', ' max_positive_polarity', ' min_negative_polarity'
             , ' max_negative_polarity', ' avg_positive_polarity', ' avg_negative_polarity']
             , axis = 1, inplace = True)
##-- Further removed correlated variables to avoid multicollinearity  
##-------------------------------------------------------------------------------------------------------

## In order to choose which threshold to use to classify as popular vs non-popular, we created a cumulative frequency plot
## and see where the distance between total shares of each article starts to become farther apart. 

def cdf(x):
        x = np.sort(x)
        n = len(x)
        y = np.arange(1, n + 1, 1)/n
        return x, y 
    
x, y = cdf(df[' shares'])

f, ax = plt.subplots(figsize=(9,4))
ax.plot(x,y,'.', color = "orange", alpha = .3)
ax.set_xlim(0,5000)
f

## setting the threshold for the binary 'y' predictor variable & standardizing the data 
df['y'] = np.where((df[' shares'] >= 3000), 1, 0)
df.drop(' shares', axis = 1, inplace = True)
df.loc[:, df.dtypes == np.float64] = StandardScaler().fit_transform(df.loc[:, df.dtypes == np.float64])
##-- We also created the predictor 'y' variable based on the threshold we got from the propensity graph
##-- we saw earlier, and standardized the data for prediction purposes.
##-------------------------------------------------------------------------------------------------------

## The following PCA/Clustering     code is exploratory analysis that was not part of our main analysis - 
## this is not included in our slides

##-------------------------------------------------------------------------------------------------------------------

## PCA & Clustering
pca = PCA(n_components = 10)
pComp = pca.fit_transform(df)
pComp = pd.DataFrame(data = pComp, columns = ['pca 1', 'pca 2', 'pca 3', 'pca 4', 'pca 5',
                                              'pca 6', 'pca 7', 'pca 8', 'pca 9', 'pca 10'])

pComp = pd.concat([pComp, df['y']], axis = 1)
pComp = pd.concat([pComp, df_main[' shares']], axis = 1)
bins = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,9000,
        10000,max(df_main[' shares'])]
pComp['split'] = pd.cut(df_main[' shares'], bins)

# graph 1
plt.figure(figsize=(15,8))
sns.scatterplot(x = 'pca 1', y = 'pca 2', data = pComp, hue = 'y')

# graph 2
plt.figure(figsize=(15,8))
sns.scatterplot(x = 'pca 1', y = 'pca 2', data = pComp, hue = 'split')

# graph 3
plt.figure(figsize=(15,8))
sns.scatterplot(x = 'pca 1', y = 'pca 2', data = pComp[pComp[' shares'] >= 10000], hue = 'split')

##-- We created 10 PCAs, PCA 1 & 2 consisted 12% & 10% of variation approximately. We can see in the first
##-- 2 graphs, there isn't a clear distinction between y or the bins. They all overlap eachother. Even in shares
##-- over 10000, the observation points don't follw any real trend. We can create other graphs as well
##-------------------------------------------------------------------------------------------------------

## Time to model, Logistic Regression
# initiating Logistic Regression
log_reg = LogisticRegression(C = 5) 

# Separating dependent & independent variables
X = df.iloc[:,0:30]
y = df.iloc[:,30]
y = pd.Series(y)

# Splitting for test & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)   
# fitting the regression on our training data 
log_reg.fit(X_train, y_train)
# predictions on our testing data
y_pred = log_reg.predict_proba(X_test)  

# Getting the top 10 coefficients
coef = pd.DataFrame(log_reg.coef_).T
coef = pd.concat([pd.DataFrame(X_test.columns),coef], axis = 1)
coef.columns = ['name', 'coef']
coef.nlargest(10,'coef')

# getting the accuracy measures
y_pred1 = pd.DataFrame(y_pred[:,0])
y_pred1['class'] = np.where((y_pred1[0] >= .75), 0, 1)
print(confusion_matrix(y_test,y_pred1['class']))
print(classification_report(y_test,y_pred1['class']))  
print(accuracy_score(y_test, y_pred1['class']))  

# checking accuracy at multiple threshold of probability output
threshold = [0.10,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
acc = []
for i in threshold:
    y_pred1 = pd.DataFrame(y_pred[:,0])
    y_pred1['class'] = np.where((y_pred1[0] >= i), 0, 1)
    acc.append(accuracy_score(y_test, y_pred1['class']))  

# getting the AUC curve
sns.set('talk', 'whitegrid', 'dark', font_scale=1.5, font='Ricty',
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

fpr, tpr, _ = roc_curve(y_test, y_pred[:,1])
roc_auc = auc(fpr, tpr)

lw = 2
plt.figure(figsize=(15,8))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()
##-------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------

## Exploratory stuff
sns.barplot(x = " n_tokens_title", y = " shares", data = df_main)
plt.show()

sns.barplot(x = " title_sentiment_polarity", y = " shares", data = df_main)
plt.show()

a = df_main[(df[' shares'] <= 25000) & (df[" n_tokens_content"] <= 4000)]
sns.regplot(x = " n_tokens_content", y = ' shares', data = a, scatter_kws={'alpha':0.2})
plt.show()

a = df_main.loc[(df[' shares'] <= 10000) & (df[" n_tokens_content"] <= 4000), [" n_tokens_title", ' shares']]
sns.boxplot(x = " n_tokens_title", y = ' shares', data = a)
plt.show()

a = df_main.loc[(df[' n_tokens_content'] <= 2500)]
sns.boxplot(x = " n_tokens_title", y = " n_tokens_content", data = a)

a = np.array(df[' shares'])
y = sns.kdeplot(a)
y.set_xlim(0,50000)
##-------------------------------------------------------------------------------------------------------

## Other Methods Tried
###--------- Support Vector Machines ---------###
svclassifier = SVC(kernel='rbf', C = 1, gamma = 100)  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  
y_pred1['class'] = np.where((y_pred1 >= 0.25), 0, 1)

print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))  
print(accuracy_score(y_test, y_pred1))  

###--------- Random Forest ---------###
regressor = RandomForestRegressor(n_estimators = 50, min_samples_split = 7, random_state = 0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)  

y_pred1 = pd.DataFrame(y_pred)
y_pred1['class'] = np.where((y_pred1 >= 0.25), 0, 1)

print(confusion_matrix(y_test,y_pred1['class']))  
print(classification_report(y_test,y_pred1['class']))  
print(accuracy_score(y_test, y_pred1['class']))  

##-- We tried other methids as well. But they did not do too well for us. Even though the data is rich in 
##-- information, we suspect it is more skewed towards prediciting the 0s, we saw that in both RandomForest
##-- and SVM. In terms of trying different parameters and getting the best balance for our predictions,
##-- both SVM and Random Forest strongly skewed to either 1 or 0. Whereas Logistic Rgression gave us the
##-- best balance.
##-------------------------------------------------------------------------------------------------------
















