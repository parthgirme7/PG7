# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:14:58 2018

@author: ParthGirme7
"""

import pandas as pd
import numpy as np

df = pd.read_csv('OnlineNewsPopularity.csv')
df.head()
df.dtypes

df.drop(['url', ' timedelta'], axis = 1, inplace = True)
df['y'] = np.NAN
#df['y'] =  1 if df[' shares'] >= 3500 else 0
df['y'] = np.where((df[' shares'] >= 3500), 1, 0)

df1 = pd.read_csv('OnlineNewsPopularity.csv')
df1.drop(['url', ' timedelta', ' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max',' kw_max_max', ' kw_avg_max',
               ' kw_min_avg',' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares',
               ' self_reference_max_shares', ' self_reference_avg_sharess', ' is_weekend',
               ' global_subjectivity', ' title_subjectivity', ' abs_title_subjectivity'], axis = 1, inplace = True)

df_std = pd.read_csv('OnlineNewsPopularity.csv')
df_std.drop(['url', ' timedelta'], axis = 1, inplace = True)

#df = df.astype(int)
df.describe().T

for i in df:
    if df[i].nunique() <= 20:
        print(i)
        
## checking number of unique values for each var ###
for i in df1:
    print(i,(df1[i].nunique()))
    
## correlation for both datasets
nan = []
for i in df:
    if df[i].nunique() <= 20:
        nan.append(i)
        df[nan] = df[nan].astype(object)

nan1 = []
for i in df1:
    if df1[i].nunique() <= 20:
        nan1.append(i) 
        df1[nan1] = df1[nan1].astype(object)
        
a = df.corr()

aa = df1.corr()
    
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x = " title_sentiment_polarity", y = " shares", data = df)
plt.show()

#### standardization #############

for i in df1:
    if df1[i].nunique() <= 10:
        print(i,(df1[i].nunique()))

from sklearn.preprocessing import StandardScaler 

#std = []
#for i in df:
#    if df[i].dtypes == 'float64':
#        std.append[i]

df_std.loc[:, df_std.dtypes == np.float64] = StandardScaler().fit_transform(df_std.loc[:, df_std.dtypes == np.float64])

#### checking the spread of shares #############

bins = [0,2500,5000,7500,max(df[' shares'])]
df[' bins'] = pd.cut(df[' shares'], bins = bins) 
df[' bins'].value_counts()

bins = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,max(df[' shares'])]
df[' bins'] = pd.cut(df[' shares'], bins = bins) 
df[' bins'].value_counts()

bins = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,9000,10000,max(df[' shares'])]
df[' bins'] = pd.cut(df[' shares'], bins = bins) 
df[' bins'].value_counts()

df.groupby(' bins')[' bins'].count()

### PCA ######
### for whole dataframe ####

from sklearn.decomposition import PCA
df_pca = pd.read_csv('OnlineNewsPopularity.csv')
df_pca.drop(['url', ' timedelta', ' shares'], axis = 1, inplace = True)
df_pca.loc[:, df_pca.dtypes == np.float64] = StandardScaler().fit_transform(df_pca.loc[:, df_pca.dtypes == np.float64])

pca = PCA(n_components = 25)
principalComponents = pca.fit_transform(df_pca)
principalComponents = pd.DataFrame(data = principalComponents, columns = ['pca 1', 'pca 2', 'pca 3', 'pca 4', 'pca 5',
                                                                          'pca 6', 'pca 7', 'pca 8', 'pca 9', 'pca 10',
                                                                          'pca 11', 'pca 12', 'pca 13', 'pca 14', 'pca 15',
                                                                          'pca 16', 'pca 17', 'pca 18', 'pca 19', 'pca 20',
                                                                          'pca 21', 'pca 22', 'pca 23', 'pca 24', 'pca 25'])

pca.explained_variance_ratio_
print(abs( pca.components_[1] ))

bins = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,9000,10000,max(df[' shares'])]
bins = [0,3500,max(df[' shares'])]
principalComponents  = pd.concat([principalComponents, df[' shares']], axis = 1)
principalComponents['split'] = pd.cut(principalComponents[' shares'], bins)

from matplotlib import pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,8))
sns.scatterplot(x = 'pca 1', y = 'pca 2', data = principalComponents, hue = 'split')

### for selected dataframe #####

df1_pca = pd.read_csv('OnlineNewsPopularity.csv')
df1_pca.drop(['url', ' timedelta', ' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max',' kw_max_max', ' kw_avg_max',
               ' kw_min_avg',' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares',
               ' self_reference_max_shares', ' self_reference_avg_sharess', ' is_weekend',
               ' global_subjectivity', ' title_subjectivity', ' abs_title_subjectivity', ' shares'], axis = 1, inplace = True)
df1_pca.loc[:, df1_pca.dtypes == np.float64] = StandardScaler().fit_transform(df1_pca.loc[:, df1_pca.dtypes == np.float64])

pca1 = PCA(n_components = 25)
principalComponents1 = pca.fit_transform(df1_pca)
principalComponents1 = pd.DataFrame(data = principalComponents1, columns = ['pca 1', 'pca 2', 'pca 3', 'pca 4', 'pca 5',
                                                                          'pca 6', 'pca 7', 'pca 8', 'pca 9', 'pca 10',
                                                                          'pca 11', 'pca 12', 'pca 13', 'pca 14', 'pca 15',
                                                                          'pca 16', 'pca 17', 'pca 18', 'pca 19', 'pca 20',
                                                                          'pca 21', 'pca 22', 'pca 23', 'pca 24', 'pca 25'])

pca.explained_variance_ratio_.tolist()
print(abs( pca.components_[1]))

principalComponents1  = pd.concat([principalComponents1, df[' shares']], axis = 1)
principalComponents1['split'] = pd.cut(principalComponents1[' shares'], bins)

from matplotlib import pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,8))
sns.scatterplot(x = 'pca 1', y = 'pca 2', data = principalComponents1[principalComponents1[' shares'] >= 10000], hue = 'split')















