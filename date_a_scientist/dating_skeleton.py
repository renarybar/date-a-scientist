import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import nltk  # Natural Language Toolkit
from collections import Counter
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from math import sqrt


#Create your df here:
df = pd.read_csv('c:/Users/rryba/Documents/python_workspace/date_a_scientist/profiles.csv')
#print different columns data
""" print(df.head())
print(df.job.head())
print(df.columns.values)

print(len(df))
print(df.drinks.value_counts())
print(df.drugs.value_counts())
print(df.smokes.value_counts())
print(df.body_type.unique())
print(df.age.value_counts())
print(df.income.value_counts()) """

#plot histogram of age
#plt.hist(df.age, bins=20)
#plt.title('Histogram of age')
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.xlim(16, 80)
#plt.show()

#create mapping for drinks, drugs, smoke, sex, body type
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)
df["drinks_code"].fillna(-1, inplace=True)

drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drugs_mapping)
df["drugs_code"].fillna(-1, inplace=True)

smokes_mapping = {"no": 0, "when drinking": 1, "sometimes": 2, "trying to quit":3, "yes":4}
df["smokes_code"] = df.smokes.map(smokes_mapping)
df["smokes_code"].fillna(-1, inplace=True)

sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)
df["sex_code"].fillna(-1, inplace=True)

body_type_mapping = {"thin": 0, "skinny": 0, "a little extra": 0, "fit": 1, "jacked": 1, "athletic": 1,"average": 2, "curvy": 3, "full figured": 3, "used up": 3, "overweight": 3, "rather not say": 3}
df["body_type_code"] = df.body_type.map(body_type_mapping)
df["body_type_code"].fillna(-1, inplace=True)

#bin age evalues and remove outliers - more than 80years old
df["age"] = df[(df[["age"]] <= 80).all(axis=1)]
bins = [0, 20, 30, 40, 50, 60, 70]
bins_labels = [20, 30, 40, 50, 60, 70]
df["age_binned"] = pd.cut(df["age"], bins=bins, labels=bins_labels)
#s = pd.cut(df["age"], bins=bins, labels=bins_labels).value_counts()
#print(s)

#mapping age
age_mapping = {"20": 0, "30": 1, "40": 2, "50": 3, "60": 4, "70": 0}
df["age_code"] = df.age_binned.map(age_mapping)
df["age_code"].fillna(-1, inplace=True)

#essays features
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
#store it into df
df["all_essays"] = all_essays

#length of the essay
df["essay_len"] = all_essays.apply(lambda x: len(x))
#count number of I, I'm or me
df["I_count"]= all_essays.apply(lambda x: sum(s in {"i", "i'm", "me"} for s in nltk.wordpunct_tokenize(x)))
#average length of words in the essay

def avg_word(sentence):
  words = sentence.split()
  for word in words:
    if len(words) ==0:
        return 0
    if len(words)!=0:
        return (sum(len(word) for word in words)/len(words))

df["avg_word"] = all_essays.apply(lambda x: avg_word(x))

#features to show if the text was written by men or women
#count number of ! (mostly used by women)
df["exclamation_count"]= all_essays.apply(lambda x: sum(s in {"!"} for s in nltk.wordpunct_tokenize(x)))
#count number of the, a(mostly used by men)
df["articles_count"]= all_essays.apply(lambda x: sum(s in {"the", "a"} for s in nltk.wordpunct_tokenize(x)))
#count number of like, a lot of (mostly used by men)
df["quant_count"]= all_essays.apply(lambda x: sum(s in {"like", "a lot of"} for s in nltk.wordpunct_tokenize(x)))


###################Plot different features #################################

"""plt.hist(df.income, bins = 60)
plt.title('Histogram of income')
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

plt.hist(df.essay_len, bins=500)
plt.title('Histogram of essays length')
plt.xlabel("Essays length")
plt.ylabel("Frequency")
plt.xlim(0, 15000)
plt.show()


plt.scatter(df.age, df.drinks_code)
plt.title('Correlation between age and drinks')
plt.xlabel("Age")
plt.ylabel("Drinks code")
plt.show() 

plt.scatter(df.age, df.drugs_code)
plt.title('Correlation between age and drugs')
plt.xlabel("Age")
plt.ylabel("Drugs code")
plt.show()


plt.scatter(df.age, df.smokes_code)
plt.title('Correlation between age and smoking')
plt.xlabel("Age")
plt.ylabel("Smokes code")
plt.show() """


###################################################################################################################
## The following code contains different classification and regression models.                                   ##
## They are separated and I alsways tested only one model at time.                                               ##
## Therefore the names of variables canbe the same for each model, e.g. train_data, test_data,...                ## 
## Only Linear regresion and KNN for regression are together in one part of code.                                ##
###################################################################################################################


#######  Predict income from age: Linear Regression and KNN Regression######################################################
#take age and salary columns, remove all NAN
age_income = df.loc[:,['age', 'income']]   #select rows by labels/index
age_income.dropna(inplace=True)
#remove rows without specified income
age_income = age_income[(age_income[['income']] != -1).all(axis=1)]
#remove rows with income bigger than 500000
age_income = age_income[((age_income[['income']] <500000)).all(axis=1)]
feature = age_income[['age']]
income = age_income[['income']]

#number of dropped rows because of outliers
#dif = len(df.income)-len(age_income['income'])
#print(dif)
#49011

X_train, X_test, y_train, y_test = train_test_split(feature, income, test_size = 0.2, random_state = 1)

model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_predicted = model_lr.predict(X_test)

print('Regression coeficients: ', model_lr.coef_)
print('Regression intercept: ',model_lr.intercept_)

#calculate error
error = sqrt(mean_squared_error(y_test,y_predicted)) #calculate rmse
print('RMSE value= ', error)

print('Age v. Income LR Score for train set: ', model_lr.score(X_train,y_train))
print('Age v. Income LR Score for test set:: ', model_lr.score(X_test,y_test))

#graphing the results
plt.scatter(X_test,y_test,alpha=0.2)
plt.plot(X_test, y_predicted)
plt.title('Linear Regression: Income predicted form age')
plt.xlabel('Age')
plt.ylabel('Income')
plt.xlim(16,80)
plt.ylim(10000,300000)
plt.show()

#KNN (as regression) #
#create classifier KNeighbors
#use the same train and test dat aas for LR above
accuracies = []
rmse_val=[]
k_list = range (1,100)

for k in k_list:
  model_reg = KNeighborsClassifier(n_neighbors = k)
  #train classifier
  model_reg.fit(X_train, y_train)
  #find how accuracte it is by score function
  guess = model_reg.score(X_test, y_test)
  accuracies.append([guess])
  #calculate rmse for each k
  y_pred=model_reg.predict(X_test) #make prediction on test set
  error = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
  rmse_val.append(error) #store rmse values
  
print('RMSE value is:', rmse_val) 

max_y = max(accuracies)  # Find the maximum value from accuracies
max_x = k_list[accuracies.index(max_y)]  # Find the k value corresponding to the maximum accuracy value
print(max_x, max_y)

classifier_reg = KNeighborsClassifier(n_neighbors = max_x)
classifier_reg.fit(X_train, y_train)
y_KNN_predicted = classifier_reg.predict(X_test)

print('Age v. Income KNN Regression Score for train set: ', classifier_reg.score(X_train,y_train))
print('Age v. Income KNN Regression Score for test set: ', classifier_reg.score(X_test,y_test))

#graphing the results
plt.scatter(X_test,y_test,alpha=0.2)
plt.scatter(X_test, y_KNN_predicted)
plt.title('KNN Regresison: Income predicted form age')
plt.xlabel('Age')
plt.ylabel('Income')
plt.xlim(16,80)
plt.ylim(10000,300000)
plt.show()  

#find k with the smallest RMSE
min_y = min(rmse_val)  # Find the minimum value from rmse_val
min_x = k_list[rmse_val.index(min_y)]  # Find the k value corresponding to the maximum accuracy value
print(min_x, min_y)

#plotting the rmse values against k values
plt.scatter(k_list, rmse_val)
plt.title('RMSE for KNN Regression')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.show()

#KNN for the smallest RMSE, k=1
classifier_reg_min = KNeighborsClassifier(n_neighbors = min_x)
classifier_reg_min.fit(X_train, y_train)
y_KNN_predicted_min = classifier_reg_min.predict(X_test)

#graphing the results
print('Age v. Income KNN Regression Score: ', classifier_reg_min.score(X_test,y_test))
plt.scatter(X_test,y_test,alpha=0.2)
plt.scatter(X_test, y_KNN_predicted_min)
plt.title('KNN Regresison: Income predicted form age')
plt.xlabel('Age')
plt.ylabel('Income')
plt.xlim(16,80)
plt.ylim(10000,300000)
plt.show()


####### Predict age from smokes, drinks, drugs: KNN ###############################################
""" #list of features, clear all NaN
#all features
features_age = df.loc[:,['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'I_count', 'avg_word', 'age_binned']]
#only drugs, drinks and smoke
#features_age = df.loc[:,['smokes_code', 'drinks_code', 'drugs_code', 'age_binned']]   #select rows by labels/index
#only essay features
#features_age = df.loc[:,['essay_len', 'I_count', 'avg_word', 'age_binned']]
features_age.dropna(inplace=True)
features_age = features_age[(features_age[['age_binned']] != -1).all(axis=1)]
feature_data = features_age[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'I_count', 'avg_word']]
#labels 
labels = features_age['age_binned']

#normalization
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
feature_data_scaled = min_max_scaler.fit_transform(x)

#Creating the Training Set and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(feature_data_scaled, labels, test_size = 0.2, random_state = 1) 

#find optimal value for k
accuracies = []
k_list = range (1,100)

for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  #train classifier
  classifier.fit(train_data, train_labels)
  #find how accuracte it is by score function
  guess = classifier.score(test_data, test_labels)
  accuracies.append([guess])
 
#graphing the results
plt.scatter(k_list, accuracies)
plt.title('KNN Classifier Accuracy: Age predicted form drugs, drinks, smoke')
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.show()

max_y = max(accuracies)  # Find the maximum value from accuracies
max_x = k_list[accuracies.index(max_y)]  # Find the k value corresponding to the maximum accuracy value
print(max_x, max_y)

#classifier for ONE defined k - with the highest score
classifier_k = KNeighborsClassifier(n_neighbors = max_x)
classifier_k.fit(train_data, train_labels)
score = classifier_k.score(test_data, test_labels)

predictions = classifier_k.predict(test_data)
print(confusion_matrix(test_labels, predictions)) 
print(accuracy_score(test_labels, predictions))
print (recall_score(test_labels, predictions, average=None))
print(precision_score(test_labels, predictions, average=None))
print(f1_score(test_labels, predictions, average=None))  """


####### Predict age from smokes, drinks, drugs: SVC ###############################################
""" #list of features, clear all NaN
#only drugs, drinks and smoke
features_age = df.loc[:,['smokes_code', 'drinks_code', 'drugs_code', 'age_binned']]   #select rows by labels/index
features_age.dropna(inplace=True)
features_age = features_age[(features_age[['age_binned']] != -1).all(axis=1)]
feature_data = features_age[['smokes_code', 'drinks_code', 'drugs_code']]
#labels 
labels = features_age['age_binned']

#normalization
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
feature_data_scaled = min_max_scaler.fit_transform(x)

#Creating the Training Set and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(feature_data_scaled, labels, test_size = 0.2, random_state = 1) 

#find the best model parameters
clf = SVC()
parameters = {'gamma': list(range(1,20)), 'C': [0.9]}
#create a dictionary of all the parameters and values you want to search through
grid_clf = GridSearchCV(clf, param_grid=parameters)  #function goes through all defined parameters
grid_clf.fit(train_data, train_labels)

best_model = grid_clf.best_estimator_
best_model_paramters = grid_clf.best_params_
best_model_score = grid_clf.best_score_
print(best_model_paramters)
print(best_model_score) 

#create classifier with the best parameters
classifier_svc = SVC(kernel = "rbf", gamma = 10, C=0.9)

classifier_svc.fit(train_data, train_labels)
predicted = classifier_svc.predict(test_data)

print(classifier_svc.score(test_data, test_labels))
print(confusion_matrix(test_labels, predicted)) 
print (recall_score(test_labels, predicted, average=None))
print(precision_score(test_labels, predicted, average=None))
print(f1_score(test_labels, predicted, average=None))  """

####### Predict income from sex and 3 essay features: KNN #########################################
""" #list of features, clear all NaN
features_income = df.loc[:,['sex_code','essay_len', 'I_count', 'avg_word', 'income']]   #select rows by labels/index
features_income.dropna(inplace=True)
features_income = features_income[(features_income[['income']] != -1).all(axis=1)]

feature_data = features_income[['sex_code','essay_len', 'I_count', 'avg_word']]
#labels 
labels = features_income['income']

#normalization
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
feature_data_scaled = min_max_scaler.fit_transform(x)

#Creating the Training Set and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(feature_data_scaled, labels, test_size = 0.2, random_state = 1)

#find optimal value for k
accuracies = []

for k in range (1, 100):
  classifier = KNeighborsClassifier(n_neighbors = k)
  #train classifier
  classifier.fit(train_data, train_labels)
  #find how accuracte it is by score function
  guess = classifier.score(test_data, test_labels)
  accuracies.append([guess])
 	
#graphing the results
k_list = range (1,100)
plt.scatter(k_list, accuracies)
plt.title('KNN Classifier Accuracy: Income predicted form sex and essays features')
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.show()

max_y = max(accuracies)  # Find the maximum value from accuracies 
max_x = k_list[accuracies.index(max_y)]  # Find the k value corresponding to the maximum accuracy value
print(max_x, max_y)

#classifier for ONE defined k - with the highest score
classifier_k = KNeighborsClassifier(n_neighbors = max_x)
classifier_k.fit(train_data, train_labels)
score = classifier_k.score(test_data, test_labels)

predictions = classifier_k.predict(test_data)
print(confusion_matrix(test_labels, predictions)) 
print(accuracy_score(test_labels, predictions))
print (recall_score(test_labels, predictions, average=None))
print(precision_score(test_labels, predictions, average=None))
print(f1_score(test_labels, predictions, average=None))   """

####### Predict income from sex and 3 essay features: SVC ###############################################
""" #list of features, clear all NaN
features_income = df.loc[:,['sex_code','essay_len', 'I_count', 'avg_word', 'income']]   #select rows by labels/index
features_income.dropna(inplace=True)
features_income = features_income[(features_income[['income']] != -1).all(axis=1)]

feature_data = features_income[['sex_code','essay_len', 'I_count', 'avg_word']]
#labels 
labels = features_income['income']

#normalization
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
feature_data_scaled = min_max_scaler.fit_transform(x)

#Creating the Training Set and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(feature_data_scaled, labels, test_size = 0.2, random_state = 1)

#find the best model parameters
clf = SVC()
parameters = {'gamma': list(range(1,20)), 'C': [0.9]}
#create a dictionary of all the parameters and values you want to search through
grid_clf = GridSearchCV(clf, param_grid=parameters)  #function goes through all defined parameters
grid_clf.fit(train_data, train_labels)

best_model = grid_clf.best_estimator_
best_model_paramters = grid_clf.best_params_
best_model_score = grid_clf.best_score_
print(best_model_paramters)
print(best_model_score)

#create classifier with the best parameters
classifier_svc = SVC(kernel = "rbf", gamma = 1, C=0.9)

classifier_svc.fit(train_data, train_labels)
predicted = classifier_svc.predict(test_data)

print(classifier_svc.score(test_data, test_labels))
print(confusion_matrix(test_labels, predicted)) 
print (recall_score(test_labels, predicted, average=None))
print(precision_score(test_labels, predicted, average=None))
print(f1_score(test_labels, predicted, average=None)) """


####### Predict body type from drinks, drugs, smoke, sex, age: KNN ########################################
#list of features, clear all NaN
""" features_body = df.loc[:,['smokes_code', 'drinks_code', 'drugs_code','sex_code', 'age_code', 'body_type_code']]   #select rows by labels/index
features_body.dropna(inplace=True)
features_body = features_body[(features_body[['body_type_code']] != -1).all(axis=1)]

feature_data = features_body[['smokes_code', 'drinks_code', 'drugs_code','sex_code', 'age_code']]
#labels 
labels = features_body['body_type_code']

#normalization
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
feature_data_scaled = min_max_scaler.fit_transform(x)

#Creating the Training Set and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(feature_data_scaled, labels, test_size = 0.2, random_state = 1)

#find optimal value for k
accuracies = []

for k in range (1, 100):
  classifier = KNeighborsClassifier(n_neighbors = k)
  #train classifier
  classifier.fit(train_data, train_labels)
  #find how accuracte it is by score function
  guess = classifier.score(test_data, test_labels)
  accuracies.append([guess])
 	
#graphing the results
k_list = range (1,100)
plt.scatter(k_list, accuracies)
plt.title('KNN Classifier Accuracy: Body type predicted form drinks, drugs, smoke, age and sex')
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.show() 

max_y = max(accuracies)  # Find the maximum value from accuracies
max_x = k_list[accuracies.index(max_y)]  # Find the k value corresponding to the maximum accuracy value
print(max_x, max_y)

#classifier for ONE defined k - with the highest score
classifier_k = KNeighborsClassifier(n_neighbors = max_x)
classifier_k.fit(train_data, train_labels)
score = classifier_k.score(test_data, test_labels)

predictions = classifier_k.predict(test_data)
print(confusion_matrix(test_labels, predictions)) 
print(accuracy_score(test_labels, predictions))
print (recall_score(test_labels, predictions, average=None))
print(precision_score(test_labels, predictions, average=None))
print(f1_score(test_labels, predictions, average=None))  """

####### Predict body type from drinks, drugs, smoke, sex, age: SVC ########################################
""" #list of features, clear all NaN
features_body = df.loc[:,['smokes_code', 'drinks_code', 'drugs_code','sex_code', 'age_code', 'body_type_code']]   #select rows by labels/index
features_body.dropna(inplace=True)
features_body = features_body[(features_body[['body_type_code']] != -1).all(axis=1)]

feature_data = features_body[['smokes_code', 'drinks_code', 'drugs_code','sex_code', 'age_code']]
#labels 
labels = features_body['body_type_code']

#normalization
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
feature_data_scaled = min_max_scaler.fit_transform(x)

#Creating the Training Set and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(feature_data_scaled, labels, test_size = 0.2, random_state = 1)

#find the best model parameters
clf = SVC()
parameters = {'gamma': list(range(1,20)), 'C': [0.9]}
#create a dictionary of all the parameters and values you want to search through
grid_clf = GridSearchCV(clf, param_grid=parameters)  #function goes through all defined parameters
grid_clf.fit(train_data, train_labels)

best_model = grid_clf.best_estimator_
best_model_paramters = grid_clf.best_params_
best_model_score = grid_clf.best_score_
print(best_model_paramters)
print(best_model_score)

#create classifier with the best parameters
classifier_svc = SVC(kernel = "rbf", gamma = 7, C=0.9)

classifier_svc.fit(train_data, train_labels)
predicted = classifier_svc.predict(test_data)

print(classifier_svc.score(test_data, test_labels))
print(confusion_matrix(test_labels, predicted)) 
print (recall_score(test_labels, predicted, average=None))
print(precision_score(test_labels, predicted, average=None))
print(f1_score(test_labels, predicted, average=None)) """

####### Predict sex from essays: Bayse classifier ############################################
""" input_data = df.loc[:,['all_essays','sex_code']]   #select rows by labels/index
input_data.dropna(inplace=True)
input_data = input_data[(input_data[['sex_code']] != -1).all(axis=1)]

data = input_data['all_essays']
labels = input_data['sex_code']

#making a training and test set
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 1)

#making the count vectors
counter = CountVectorizer()
#This teaches the counter our vocabulary.
counter.fit(train_data)
#this creates numbers of words 
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

#Print train_data[2] and train_counts[2] to see what a tweet looks like as a Count Vector.
#print(train_data[2])
#print(train_counts[2])

#train and test the Naine Bayes Classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)

#test our model
predictions = classifier.predict(test_counts)

#evaluating the model
score = accuracy_score(test_labels, predictions)
print(score)
print(confusion_matrix(test_labels, predictions))
print (recall_score(test_labels, predictions))
print(precision_score(test_labels, predictions))
print(f1_score(test_labels, predictions))
 """

####### Predict sex from essays features: KNN #############################################
""" #list of features from essays, clear all NaN
features_age = df.loc[:,['essay_len', 'I_count', 'avg_word', 'exclamation_count', 'articles_count', 'quant_count', 'sex_code']]
features_age.dropna(inplace=True)
features_age = features_age[(features_age[['sex_code']] != -1).all(axis=1)]
feature_data = features_age[['essay_len', 'I_count', 'avg_word', 'exclamation_count', 'articles_count', 'quant_count']]
#labels 
labels = features_age['sex_code']

#normalization
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
feature_data_scaled = min_max_scaler.fit_transform(x)

#Creating the Training Set and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(feature_data_scaled, labels, test_size = 0.2, random_state = 1) 

#find optimal value for k
accuracies = []
recall = []
precision = []
f1 = []
k_list = range (1,100)
 
for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  #train classifier
  classifier.fit(train_data, train_labels)
  #find how accuracte it is by score function
  predictions = classifier.predict(test_data)
  guess = classifier.score(test_data, test_labels)
  recall_s = recall_score(test_labels, predictions)
  precision_s = precision_score(test_labels, predictions)
  f1_s = f1_score(test_labels, predictions)
  accuracies.append([guess])
  recall.append([recall_s])
  precision.append([precision_s])
  f1.append([f1_s])
 
#graphing the results
plt.scatter(k_list, accuracies)
plt.title('KNN Classifier Accuracy: Sex predicted form various essays features')
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.show()

#recall
plt.scatter(k_list, recall)
plt.title('KNN Recall: Sex predicted form various essays features')
plt.xlabel('K')
plt.ylabel('Recall')
plt.show()

#precision
plt.scatter(k_list, precision)
plt.title('KNN Precision: Sex predicted form various essays features')
plt.xlabel('K')
plt.ylabel('Precision')
plt.show()

#f1
plt.scatter(k_list, f1)
plt.title('KNN F1 score: Sex predicted form various essays features')
plt.xlabel('K')
plt.ylabel('F1 score')
plt.show()


max_y = max(accuracies)  # Find the maximum value from accuracies
max_x = k_list[accuracies.index(max_y)]  # Find the k value corresponding to the maximum accuracy value
print(max_x, max_y)

#classifier for ONE defined k - with the highest score
classifier_k = KNeighborsClassifier(n_neighbors = max_x)
classifier_k.fit(train_data, train_labels)
score = classifier_k.score(test_data, test_labels)

predictions = classifier_k.predict(test_data)
print(confusion_matrix(test_labels, predictions)) 
print(accuracy_score(test_labels, predictions))
print (recall_score(test_labels, predictions))
print(precision_score(test_labels, predictions))
print(f1_score(test_labels, predictions))  """

#classifier for ONE defined k = 42, where the score was almost the same as for k=max_x
""" classifier_k = KNeighborsClassifier(n_neighbors = 42)
classifier_k.fit(train_data, train_labels)
score = classifier_k.score(test_data, test_labels)

predictions = classifier_k.predict(test_data)
print(confusion_matrix(test_labels, predictions)) 
print(accuracy_score(test_labels, predictions))
print (recall_score(test_labels, predictions))
print(precision_score(test_labels, predictions))
print(f1_score(test_labels, predictions))  """
