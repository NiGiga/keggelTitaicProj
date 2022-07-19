import pandas as pd

training = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['PassengerId']

def clean(training):
    training = training.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
    cols = ['SibSp', 'Parch', 'Fare', 'Age']
    for col in cols:
        training[col].fillna(training[col].median(), inplace=True)

    training.Embarked.fillna('U', inplace=True)
    return  training

training = clean(training)
test = clean(test)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

cols = ['Sex', 'Embarked']
for col in cols:
    training[col] = le.fit_transform(training[col])
    test[col] = le.transform(test[col])
    print(le.classes_)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y = training['Survived']
x = training.drop('Survived', axis=1)
x_val, x_train, y_val, y_train = train_test_split(x, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train)

predictions = clf.predict(x_val)
print(accuracy_score(y_val, predictions))

submission_preds = clf.predict(test)
df = pd.DataFrame({'PassengerId':test_ids.values,
                   'Survived': submission_preds})

df.to_csv('submission.csv', index=False)
