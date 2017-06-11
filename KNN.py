from sklearn import neighbors
import pandas as pd

# Train data
df = pd.read_csv('LargeTrain.csv')

# columns
columns =  list(df.columns.values)

# data and target
X = df[ columns[0:-1] ]
Y = df[['Class']]

# create model
clf = neighbors.KNeighborsClassifier()

# fitting model
clf.fit(X, Y)

# test data
print clf.score(X, Y)
