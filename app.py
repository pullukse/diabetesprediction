import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ztest
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

working_directory = os.getcwd()
print(working_directory)
path = working_directory + '/diabetes_data_upload.csv'
df = pd.read_csv(path)


#change string into boolean
df = df.replace("No",0)
df = df.replace("Yes",1)
df = df.replace("Negative",0)
df = df.replace("Positive",1)

#replace genders to isMale 
df = df.replace("Male",1)
df = df.replace("Female",0)

#replace column name
replace = {"Gender":"isMale",}
df = df.rename(columns=replace)
df.columns = df.columns.str.lower()

#read cleaned data
df.to_csv("diabetes_data_clean.csv", index=None)
pd.read_csv("diabetes_data_clean.csv")

#class-obesity
obesity_diabetes_ctab = pd.crosstab(df['class'],df['obesity'])
obesity_diabetes_ctab
chi2_contingency(obesity_diabetes_ctab)

#class-ismale
ismale_diabetes_ctab = pd.crosstab(df['class'],df['ismale'])
ismale_diabetes_ctab
chi2_contingency(ismale_diabetes_ctab)

#class-polyuria
polyuria_diabetes_ctab = pd.crosstab(df['class'],df['polyuria'])
polyuria_diabetes_ctab
chi2_contingency(polyuria_diabetes_ctab)

#ismale-polyuria
polyuria_ismale_ctab = pd.crosstab(df['ismale'],df['polyuria'])
polyuria_ismale_ctab
chi2_contingency(polyuria_ismale_ctab)

#median of age in diabetes & non-diabetes groups
no_diabetes = df[df['class'] == 0]
no_diabetes['age'].median()
diabetes = df[df['class'] == 1]
diabetes['age'].median()

ztest(diabetes['age'],no_diabetes['age'])

#trainings
X = df.drop('class', axis = 1)
Y = df['class']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y)

#dummy
dummy = DummyClassifier()
dummy.fit(X_train, Y_train)
dummy_pred = dummy.predict(X_test)
confusion_matrix(Y_test, dummy_pred)

#logregression
logr = LogisticRegression(max_iter=10000)
logr.fit(X_train, Y_train)
logr_pred = logr.predict(X_test)
confusion_matrix(Y_test, logr_pred)

#desctree
tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
tree_pred = tree.predict(X_test)
confusion_matrix(Y_test, tree_pred)

#forest
forest = RandomForestClassifier()
forest.fit(X_train, Y_train)
forest_pred = forest.predict(X_test)
confusion_matrix(Y_test, forest_pred)

#list each attribute based on trained importance to diabetes positive
forest.feature_importances_
pd.DataFrame({'Feature':X.columns, 'Importance':forest.feature_importances_}).sort_values('Importance', ascending = False)



