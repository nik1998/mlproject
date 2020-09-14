import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('marketing_campaign_train.csv')
s = (data.dtypes == 'object')
object_cols = list(s[s].index)
features = ['Unnamed: 0','Id','профессия', 'семейное_положение', 'образование', 'взаимодействие', 'контакт_день_недели', 'контакт_прошлый_исход']
data.drop(features, axis='columns', inplace=True)
y = data["контакт_исход"]
data.drop(["контакт_исход"], axis='columns', inplace=True)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
m = data.median()

for i in m.index:
    data[i].fillna(m[i], inplace = True)
X_train, X_valid, y_train, y_valid = train_test_split(data, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

model.fit(X_train, y_train)
melb_preds = model.predict(X_valid)
print(mean_absolute_error(y_valid, melb_preds))

X_test = pd.read_csv('marketing_campaign_test.csv')
features = ['Unnamed: 0','профессия', 'семейное_положение', 'образование', 'взаимодействие', 'контакт_день_недели', 'контакт_прошлый_исход']
X_test.drop(features, axis='columns', inplace=True)
for i in m.index:
    X_test[i].fillna(m[i], inplace = True)

predictions = model.predict(X_test)

output = pd.DataFrame({ 'Out': predictions})
output.to_csv('my_submission.csv', index=False)

f = open("out.txt",'w')
for i in predictions:
    f.write(str(i)+"\n")
f.close()
print(predictions)


