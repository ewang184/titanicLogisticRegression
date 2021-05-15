import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import re
import seaborn as sns
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df = pd.read_csv("./trainingData/train.csv")

#Create model for predictions

#clean data
cleanDf = df

#female is replaced with 1, male with 0
gender = cleanDf['Sex']
female = gender.str.contains("female")
cleanDf['Sex'] = np.where(female, 1, 0)

#C,S,Q are replaced with numerical values 1,2,3 respectively
Embark = cleanDf['Embarked']
cList = Embark.str.contains("C")
sList = Embark.str.contains("S")
cleanDf['Embarked'] = np.where(cList, 1, np.where(sList, 2, 3))
#fill empty Embarked
cleanDf.Embarked.fillna(cleanDf.Embarked.mode(), inplace = True)

#fill empty Age
cleanDf['Salutation'] = cleanDf.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
grp = cleanDf.groupby(['Salutation','Pclass'])
ageMed = grp.Age.median()

missingAgeList = []
for ind in cleanDf.index:
    Title = cleanDf['Salutation'][ind]
    Class = cleanDf['Pclass'][ind]
    missingAgeList.append(ageMed[Title, Class])

for ind in cleanDf.index:
    age = cleanDf['Age'][ind]
    if math.isnan(age):
        cleanDf['Age'][ind] = missingAgeList[ind]

#correct fare

#drop columns
cleanDf = cleanDf.drop(columns = ['Cabin'])
cleanDf = cleanDf.drop(columns = ['Name'])
cleanDf = cleanDf.drop(columns = ['Ticket'])
cleanDf = cleanDf.drop(columns = ['Salutation'])


#get model
x = cleanDf.drop(columns = ['Survived'])
x = x.to_numpy()
y = cleanDf["Survived"].values
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x, y)



#get test dataframe
dft = pd.read_csv("./testData/test.csv")

#clean data
cleanDf = dft

#female is replaced with 1, male with 0
gender = cleanDf['Sex']
female = gender.str.contains("female")
cleanDf['Sex'] = np.where(female, 1, 0)

#C,S,Q are replaced with numerical values 1,2,3 respectively
Embark = cleanDf['Embarked']
cList = Embark.str.contains("C")
sList = Embark.str.contains("S")
cleanDf['Embarked'] = np.where(cList, 1, np.where(sList, 2, 3))
#fill empty Embarked
cleanDf.Embarked.fillna(cleanDf.Embarked.mode(), inplace = True)

#fill empty Age
cleanDf['Salutation'] = cleanDf.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
grp = cleanDf.groupby(['Salutation', 'Pclass'])
ageMed = grp.Age.median()

missingAgeList = []
for ind in cleanDf.index:
    Title = cleanDf['Salutation'][ind]
    Class = cleanDf['Pclass'][ind]
    if math.isnan(ageMed[Title,Class]):
        missingAgeList.append(cleanDf['Age'].median())
    else:
        missingAgeList.append(ageMed[Title, Class])

for ind in cleanDf.index:
    age = cleanDf['Age'][ind]
    if math.isnan(age):
        cleanDf['Age'][ind] = missingAgeList[ind]

#correct fare

cleanDf['Fare'] = cleanDf['Fare'].fillna(cleanDf['Fare'].median())

#drop columns
cleanDf = cleanDf.drop(columns = ['Cabin'])
cleanDf = cleanDf.drop(columns = ['Name'])
cleanDf = cleanDf.drop(columns = ['Ticket'])
cleanDf = cleanDf.drop(columns = ['Salutation'])

testx = cleanDf
lived = model.predict(testx)
passId = cleanDf['PassengerId'].tolist()
result = pd.DataFrame(passId, columns = ['PassengerId'])
result['Survived'] = lived

result.to_csv(r'./result/submission.csv', index = False)
