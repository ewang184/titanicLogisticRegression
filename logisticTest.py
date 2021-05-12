import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import re

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df = pd.read_csv("./trainingData/train.csv")

#Create model for predictions

#clean data
cleanDf = df.drop(columns = ['Name'])
cleanDf = cleanDf.drop(columns = ['Ticket'])
cleanDf = cleanDf.drop(columns = ['Cabin'])
cleanDf = cleanDf.dropna()

#female is replaced with 1, male with 0
gender = cleanDf['Sex']
female = gender.str.contains("female")
cleanDf['Sex'] = np.where(female, 1, 0)

#C,S,Q are replaced with numerical values 1,2,3 respectively
Embark = cleanDf['Embarked']
cList = Embark.str.contains("C")
sList = Embark.str.contains("S")
cleanDf['Embarked'] = np.where(cList, 1, np.where(sList, 2, 3))

#get model
x = cleanDf.drop(columns = ['Survived'])
x = x.to_numpy()
y = cleanDf["Survived"].values
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x, y)







dft = pd.read_csv("./testData/test.csv")

#clean data
testDf = dft.drop(columns = ['Name'])
testDf = testDf.drop(columns = ['Ticket'])
testDf = testDf.drop(columns = ['Cabin'])
testDf = testDf.dropna()

#female is replaced with 1, male with 0
gender = testDf['Sex']
female = gender.str.contains("female")
testDf['Sex'] = np.where(female, 1, 0)

#C,S,Q are replaced with numerical values 1,2,3 respectively
Embark = testDf['Embarked']
cList = Embark.str.contains("C")
sList = Embark.str.contains("S")
testDf['Embarked'] = np.where(cList, 1, np.where(sList, 2, 3))

#get testing data
testx = testDf.to_numpy()

print(model.predict(testx))

