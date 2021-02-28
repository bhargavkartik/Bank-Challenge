import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
from matplotlib import pyplot as plt

# Loading the data
df = pd.read_csv('campaign-data.csv')

# Print the first 5 rows
print(df.head())

# Check if there are any null values
print(df.isna().nunique())
print(df.isnull().any())

### categorical features

# 1. job
# 2. marital
# 3. education
# 4. mortgage

### numerical features

# 1. empvarrate
# 2. conspriceidx
# 3. consconfidx
# 4. euribor3m
# 5. nremployed
# 6. age
# 7. children

# One hot encoding all the categorical features
df_job = pd.get_dummies(df.job, prefix='job')
df_marital = pd.get_dummies(df.marital, prefix='marital')
df_education = pd.get_dummies(df.education, prefix='education')
df_mortgage = pd.get_dummies(df.mortgage, prefix='mortgage')

# Concatenating the new one-hot encoded features to the original dataframe
df_preprocessed = pd.concat([df, df_job, df_marital, df_education, df_mortgage], axis=1)

# Removing the original categorical features which are one-hot encoded
categorical_columns = ['job', 'marital', 'education', 'mortgage']
df_preprocessed.drop(df_preprocessed[categorical_columns], axis=1, inplace=True)

# Renaming job_admin. to job_admin
df_preprocessed.rename(columns={'job_admin.': 'job_admin'}, inplace=True)

# Print the first 5 rows
print(df_preprocessed.head())

# Scaling the numeric features
ss = StandardScaler()

numeric_columns = ['empvarrate', 'conspriceidx', 'consconfidx', 'euribor3m', 'nremployed', 'age', 'children']
data = ss.fit_transform(df_preprocessed[numeric_columns])

# convert the array back to a dataframe
df_scaled = pd.DataFrame(data)
df_scaled.columns = numeric_columns

# Assign the scaled columns back to the df_preprocessed dataframe
df_preprocessed[numeric_columns] = df_scaled[numeric_columns]

# Label encoding the output feature 'y'

le = LabelEncoder()
le.fit(['yes', 'no'])
le_y = le.transform(df_preprocessed['y'].tolist())

# Label encoded output variable y
df_y = pd.DataFrame(le_y, columns=['y'])

# Train Test Val Split

# Creating df_X and df_y dataframe which represent the input and the output respectively.
df_X = df_preprocessed.drop(df_preprocessed[['y']], axis=1, inplace=False)

X_train_all, X_test, y_train_all, y_test = train_test_split(df_X, df_y, train_size=0.8, stratify=df_y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, train_size=0.8, stratify=y_train_all,
                                                  random_state=0)

# Building the Decision tree classifier model
depth = []

# Iterating over different values of min_samples_leaf
for i in range(2, 50):
    clf = DecisionTreeClassifier(min_samples_leaf=i)
    clf.fit(X_train, y_train)

    # perform 10 fold cross validation
    accuracy = cross_val_score(estimator=clf, X=X_val, y=y_val, cv=10).mean()
    print(i, accuracy)
    depth.append((i, accuracy.mean()))

print('------------------------------------------')

print(depth)

maximum = 0
optimal_min_leaf_value = 0
answer = None

for i in depth:
    if i[1] > maximum:
        maximum = i[1]
        optimal_min_leaf_value = i[0]
        answer = i

print(answer)
print("Optimal min leaf value", optimal_min_leaf_value)

# we choose a max_depth value of 3 so that a simpler tree is rendered.
classifier = DecisionTreeClassifier(min_samples_leaf=optimal_min_leaf_value, max_depth=3)
print('classifier == ', classifier)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Max depth = ", classifier.get_depth())

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

print('ROC-AUC = ', roc_auc)

acc_score = accuracy_score(y_test, y_pred)
print('Accuracy score = ', acc_score)

# Plot the Decision tree
fig = plt.figure(figsize=(12, 8))
_ = tree.plot_tree(classifier,
                   filled=True)

plt.show()

# Plot the Confusion Matrix
plot_confusion_matrix(classifier, X_test, y_test)
plt.show()
