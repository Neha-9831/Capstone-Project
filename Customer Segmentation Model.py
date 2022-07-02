import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
import utils

##############################################################
# Read Data 
##############################################################

# read dataset
print("\n*** Read Data ***")
df = pd.read_excel(".\data\Capstone V3.xlsx")
print("Done ...")

##############################################################
# Exploratory Data Analysis
##############################################################

# Dimensions rows & cols
print("\n*** Rows & Cols ***")
print("Rows",df.shape[0])
print("Cols",df.shape[1])

# columns
print("\n*** Column Names ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# data types
print("\n*** Data Types ***")
print(df.dtypes)

#Check for Null Values
print('\n*** Columns With Nulls ***')
print(df.isnull().sum())

###################################################
#Transformation
##################################################

colNames=['InvoiceNo']
for InvoiceNo in colNames: 
    df['InvoiceNo']= pd.to_numeric(df['InvoiceNo'], errors="coerce")
print("Done.....")

# data types
print("\n*** Data Types ***")
print(df.dtypes)

#Check for Null Values 
print('\n*** Columns With Nulls ***')
print(df.isnull().sum())

#Drop the Rows with Null Values 
df = df.dropna(subset=['InvoiceNo'])
df = df.dropna(subset=['CustomerID'])
df = df.dropna(subset=['Description'])

#Check for Null Values 
print(df.isnull().sum())

#Check the Unique Values
print("\n*** Unique Values ***")
print(df.apply(lambda x: x.nunique()))

#Converting into Date format 
df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate)

print("\n*** Head ***")
print(df.head())

# Adding the column of Total Amount 
df["Total Amount"]=df["Quantity"]*df["UnitPrice"]
df.head()


# Dimensions rows & cols
print("\n*** Rows & Cols ***")
print("Rows",df.shape[0])
print("Cols",df.shape[1])

#printing head
print("\n*** Head ***")
print(df.head())

#Consider only rows which have quantity more than 1 
df1=df[df['Quantity'] >= 1]
print(df1.head())

# Dimensions of rows & cols
print("\n*** Rows & Cols ***")
print("Rows",df1.shape[0])
print("Cols",df1.shape[1])

print("\n*** Head ***")
df1.head()

###############################################
#RFM Model
###############################################

#Recency 
df1_recency=df1.groupby(by='CustomerID',as_index=False)['InvoiceDate'].max()
df1_recency.columns=['CustomerID', 'LastPurchaseDate']
recent_date=df1_recency['LastPurchaseDate'].max()
df1_recency['Recency'] =df1_recency['LastPurchaseDate'].apply(lambda x: (recent_date - x).days)
df1_recency.head()

#Frequency
frequency_df1=df1.drop_duplicates().groupby(by=['CustomerID'], as_index=False)['InvoiceDate'].count()
frequency_df1.columns=['CustomerID', 'Frequency']
frequency_df1.head()

#Monetary
monetary_df1=df1.groupby(by='CustomerID', as_index=False)['Total Amount'].sum()
monetary_df1.columns=['CustomerID', 'Monetary']
monetary_df1.head()

df1_quantity=df1.groupby(by='CustomerID',as_index=False)['Quantity'].sum()
df1_amount=df1.groupby(by='CustomerID',as_index=False)['Total Amount'].sum()

#Clubing all the columns 
rf_df=df1_recency.merge(frequency_df1, on='CustomerID')
rfm_df=rf_df.merge(monetary_df1, on='CustomerID')
rfm_df=rfm_df.merge(df1_quantity, on='CustomerID')
rfm_df=rfm_df.merge(df1_amount, on='CustomerID')
rfm_df.head ()

#Drop Columns- Last Purchase Date 
print("\n*** Drop Cols ***")
rfm_df = rfm_df.drop('LastPurchaseDate', axis=1)
print("DONE ...")

print("\n*** Head ***")
rfm_df.head ()

#Calculating RFM Score

rfm_df['R_rank'] =rfm_df['Recency'].rank(ascending=False)
rfm_df['F_rank'] =rfm_df['Frequency'].rank(ascending=True)
rfm_df['M_rank'] =rfm_df['Monetary'].rank(ascending=True)
 
# normalizing the rank of the customers
rfm_df['R_rank_norm'] =(rfm_df['R_rank']/rfm_df['R_rank'].max())*100
rfm_df['F_rank_norm'] =(rfm_df['F_rank']/rfm_df['F_rank'].max())*100
rfm_df['M_rank_norm'] =(rfm_df['F_rank']/rfm_df['M_rank'].max())*100
 
print("\n*** Head ***")
rfm_df.head()

#Considering Frequency less than 200 to avoid outliers 
rfm_df=rfm_df[rfm_df['Frequency'] <= 200]
print(rfm_df.head())

#RFM Score
rfm_df['RFM_Score'] = 0.15 * rfm_df['R_rank_norm']+ 0.28* rfm_df['F_rank_norm'] + 0.57 * rfm_df['M_rank_norm']
rfm_df['RFM_Score'] *= 0.05
rfm_df=rfm_df.round(2)

##############################################
# Customer Loyalty based upon the Frequency of the visits
##############################################
#frequency >141 : Diamond
#140 >rfm score >101 : Platinum
# 100>frequency >51 : Gold
# 50>frequency>21 : Silver
# frequency <20 :Bronze

rfm_df["Customer_Loyalty"] =np.where(rfm_df['Frequency'] > 141 , "Diamond", (np.where( rfm_df['Frequency'] >101, "Platinum", (np.where(rfm_df['Frequency'] >51,"Gold",np.where(rfm_df['Frequency'] >21 ,'Silver', 'Bronze'))))))
rfm_df1= rfm_df[['CustomerID','RFM_Score', 'Customer_Loyalty']]

#Check the data type
print("\n*** Data Types ***")
print(rfm_df1.dtypes)

#Converting Float into Integer
rfm_df1['CustomerID']=rfm_df1['CustomerID'].astype(int)

print("\n*** Head ***")
rfm_df1.head()

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(rfm_df1))


#Pie Chart for Customer Loyalty
plt.pie(rfm_df1.Customer_Loyalty.value_counts(),
        labels=rfm_df1.Customer_Loyalty.value_counts().index,
        autopct='%.0f%%')
plt.show()


################################
# Prepare Data
################################

print("\n*** Rows & Cols ***")
print("Rows",rfm_df1.shape[0])
print("Cols",rfm_df1.shape[1])

# columns
print("\n*** Column Names ***")
print(rfm_df1.columns)

# info
print("\n*** Structure ***")
print(rfm_df1.info())

# data types
print("\n*** Data Types ***")
print(rfm_df1.dtypes)

print('\n*** Columns With Nulls ***')
print(rfm_df1.isnull().sum())


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
rfm_df1.Customer_Loyalty = le.fit_transform(rfm_df1.Customer_Loyalty)
rfm_df1.head

rfm_df1 = rfm_df1.fillna(0)
print('\n*** Columns With Nulls ***')
print(rfm_df1.isnull().sum())

# split into data & target
print("\n*** Prepare Data ***")
allCols = rfm_df1.columns.tolist()
print(allCols)

#allCols.remove(clsVars)
print(allCols)
X = rfm_df1[allCols].values
# y = df[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
#print(y.shape)
print(type(X))
#print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
#print(y[0:4])

################################
# Knn Clustering
###############################

# imports
from sklearn.cluster import KMeans

print("\n*** Compute WCSSE ***")
vIters = 20
lWcsse = []
for i in range(1, vIters):
    kmcModel = KMeans(n_clusters=i)
    kmcModel.fit(X)
    lWcsse.append(kmcModel.inertia_)
for vWcsse in lWcsse:
    print(vWcsse)

# plotting the results onto a line graph, allowing us to observe 'The elbow'
print("\n*** Plot WCSSE ***")
plt.figure()
plt.plot(range(1, vIters), lWcsse)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSSE') #within cluster sum of squares error
plt.show()

# programatically
print("\n*** Find Best K ***")
import kneed
kl = kneed.KneeLocator(range(1, vIters), lWcsse, curve="convex", direction="decreasing")
vBestK = kl.elbow
print(vBestK)

# k means cluster model
print("\n*** Model Create & Train ***")
model = KMeans(n_clusters=3, random_state=707)
model.fit(X)


######################################################
# result
######################################################

print("\n*** Model Results ***")
print(model.labels_)
rfm_df1['PredKnn'] = model.labels_

# counts for knn
print("\n*** Counts For Knn ***")
print(rfm_df1.groupby(rfm_df1['PredKnn']).size())


# class count plot
print("\n*** Distribution Plot - KNN ***")
plt.figure()
sns.countplot(data=rfm_df1, x='PredKnn', label="Count")
plt.title('Distribution Plot - KNN')
plt.show()


################################
# Hierarchical Clustering
###############################

# linkage
print("\n*** Linkage Method ***")
from scipy.cluster import hierarchy as hac
vLinkage = hac.linkage(rfm_df1, 'ward')
print("Done ...")

 # make the dendrogram
print("\n*** Plot Dendrogram ***")
print("Looks Cluttered")
plt.figure(figsize=(8,8))
hac.dendrogram(vLinkage, orientation='left')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Linkage (Ward)')
plt.show


################################
# Agglomerative Clustering
###############################

# create cluster model
print("\n*** Agglomerative Clustering ***")
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
# train and group together
lGroups = model.fit_predict(rfm_df1)
print(lGroups)
# update data frame
rfm_df1['PredHeir'] = lGroups
print("Done ...")

# counts for heir
print("\n*** Counts For Heir ***")
print(rfm_df1.groupby(rfm_df1['PredHeir']).size())

# class count plot
print("\n*** Distribution Plot - Heir ***")
plt.figure(),
sns.countplot(data=rfm_df1, x='PredHeir', label="Count")
plt.title('Distribution Plot - Heir')
plt.show()


################################
# Compare
###############################

# counts for knn
print("\n*** Counts For Knn ***")
print(rfm_df1.groupby(rfm_df1['PredKnn']).size())

# counts for heir
print("\n*** Counts For Heir ***")
print(rfm_df1.groupby(rfm_df1['PredHeir']).size())


# imports
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import v_measure_score
# confusion matrix
from sklearn.metrics import confusion_matrix
print("\n*** Confusion Matrix - Actual ***")
cm = confusion_matrix(rfm_df1['PredKnn'], rfm_df1['PredKnn'])
print(cm)
print("\n*** Confusion Matrix - Clustered ***")
cm = confusion_matrix(rfm_df1['PredKnn'], rfm_df1['PredHeir'])
print(cm)
# accuracy- accuracy 85% and above is acceptable when comparing 2 models
print("\n*** Accuracy ***")
ac = accuracy_score(rfm_df1['PredKnn'], rfm_df1['PredHeir'])*100
print(ac)

# v-measure score
print("\n*** V-Score ***")
vm = v_measure_score(rfm_df1['PredKnn'], rfm_df1['PredHeir'])
print(vm)

print("\n*** Rows & Cols ***")
print("Rows",rfm_df1.shape[0])
print("Cols",rfm_df1.shape[1])

##############################################################
# Class Variable & Counts
##############################################################

rfm_df1.head()

print("\n*** Drop Cols ***")
rfm_df1 = rfm_df1.drop('PredHeir', axis=1)
print("DONE ...")

# store class variable  
# change as required
clsVars = "PredKnn"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(rfm_df1.groupby(rfm_df1[clsVars]).size())

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(rfm_df1))

rfm_df1.head()

################################
# Classification 
# set X & y
###############################

# split into data & target
print("\n*** Prepare Data ***")
allCols = rfm_df1.columns.tolist()
print(allCols)
allCols.remove(clsVars)
print(allCols)
X = rfm_df1[allCols].values
y = rfm_df1[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
print(y[0:4])

# counts
print("\n*** Counts ***")
print(rfm_df1.groupby(rfm_df1[clsVars]).size())

# counts
print("\n*** Counts ***")
unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))


################################
# Classification - init models
###############################

# original
# import all model & metrics
print("\n*** Importing Models ***")
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
from sklearn.ensemble import BaggingClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
from sklearn.ensemble import GradientBoostingClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# https://xgboost.readthedocs.io/en/latest/
#import xgboost as xgb
print("Done ...")

# create a list of models so that we can use the models in an iterstive manner
print("\n*** Creating Models ***")
lModels = []
lModels.append(('SVM Bagging    ', BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=707)))
lModels.append(('Random Forest  ', RandomForestClassifier(random_state=707)))
#lModels.append(('XGBoosting     ', xgb.XGBClassifier(booster='gbtree', objective='multi:softprob', verbosity=0, seed=707)))
lModels.append(('AdaBoosting    ', AdaBoostClassifier(n_estimators=100, random_state=707)))
lModels.append(('GradBoosting   ', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=707)))
lModels.append(('SVM-Classifier ',  SVC(random_state=707)))
lModels.append(('KNN-Classifier ', KNeighborsClassifier()))
lModels.append(('LogRegression  ', LogisticRegression(random_state=707)))
lModels.append(('DecisionTree   ', DecisionTreeClassifier(random_state=707)))
lModels.append(('NaiveBayes     ', GaussianNB()))
for vModel in lModels:
    print(vModel)
print("Done ...")


################################
# Classification - cross validation
###############################

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModNames = []
xvAccuracy = []
xvSDScores = []
print("Done ...")

# cross validation
from sklearn import model_selection
#print("\n*** Cross Validation ***")
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # select xv folds
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=707)
    # actual corss validation
    cvAccuracy = model_selection.cross_val_score(oModelObj, X, y, cv=kfold, scoring='accuracy')
    # prints result of cross val ... scores count = lfold splits
    print(vModelName,":  ",cvAccuracy)
    # update lists for future use
    xvModNames.append(vModelName)
    xvAccuracy.append(cvAccuracy.mean())
    xvSDScores.append(cvAccuracy.std())
    
# cross val summary
print("\n*** Cross Validation Summary ***")
# header
msg = "%10s: %10s %8s" % ("Model   ", "xvAccuracy", "xvStdDev")
print(msg)
# for each model
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%10s: %5.7f %5.7f" % (xvModNames[i], xvAccuracy[i], xvSDScores[i])
    print(msg)

# find model with best xv accuracy & print details
print("\n*** Best XV Accuracy Model ***")
xvIndex = xvAccuracy.index(max(xvAccuracy))
print("Index      : ",xvIndex)
print("Model Name : ",xvModNames[xvIndex])
print("XVAccuracy : ",xvAccuracy[xvIndex])
print("XVStdDev   : ",xvSDScores[xvIndex])
print("Model      : ",lModels[xvIndex])


################################
# Classification 
# Split Train & Test
###############################

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.2, random_state=707)

# shapes
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))


################################
# Classification- Create Model
###############################

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[xvIndex])
print("Model   :", lModels[xvIndex]) 

# classifier object
# select model with best acc
print("\n*** Classfier Object ***")
model = lModels[xvIndex][1]
print(model)
# fit the model
model.fit(X_train,y_train)
print("Done ...")


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

# classifier object
print("\n*** Predict Test ***")
# predicting the Test set results
p_test = model.predict(X_test)            # use model ... predict
print("Done ...")

# accuracy
from sklearn.metrics import accuracy_score
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_test, p_test)*100
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
from sklearn.metrics import confusion_matrix
print("\n*** Confusion Matrix - Original ***")
cm = confusion_matrix(y_test, y_test)
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
print("\n*** Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_test, p_test)
print(cm)

# classification report
from sklearn.metrics import classification_report
print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)


# make dftest
# only for show
# not to be done in production
print("\n*** Recreate Test ***")
dfTest =  pd.DataFrame(data = X_test)
dfTest.columns = allCols
dfTest[clsVars] = y_test
dfTest['Predict'] = p_test
#dfTest[clsVars] = le.inverse_transform(dfTest[clsVars])
#dfTest['Predict'] = le.inverse_transform(dfTest['Predict'])
print("Done ...")

################################
# save model & vars as pickle icts
###############################

# classifier object
# select best cm acc ... why
print("\n*** Classfier Object ***")
model = lModels[xvIndex][1]
print(model)
# fit the model
model.fit(X,y)
print("Done ...")

# save model
print("\n*** Save Model ***")
import pickle
Fcap = '.\data\Capstone.pkl'
pickle.dump(model, open(Fcap, 'wb'))
print("Done ...")

# save vars as dict
print("\n*** Create Vars Dict ***")
dVars = {}
dVars['clsvars'] = clsVars
dVars['allCols'] = allCols
dVars['le'] = le
print(dVars)

# save dvars
print("\n*** Save DVars ***")
Fcap = '.\data\CapVar.pkl'
pickle.dump(dVars, open(Fcap, 'wb'))
print("Done ...")

