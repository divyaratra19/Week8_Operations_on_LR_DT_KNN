from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

# Loading the Iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Splitting the dataset into training and testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)

# Creating and training the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)

# Creating and training the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)

# Creating and training the KNN model
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)

# Confusion matrices for each model
cm_lr = confusion_matrix(y_test, lr_pred)
cm_dt = confusion_matrix(y_test, dt_pred)
cm_knn = confusion_matrix(y_test, knn_pred)

# True Positive Rate (Recall)for each model
recall_lr = recall_score(y_test, lr_pred, average='weighted')
recall_dt = recall_score(y_test, dt_pred, average='weighted')
recall_knn = recall_score(y_test, knn_pred, average='weighted')

# Precision for each model
precision_dt = precision_score(y_test, dt_pred, average='weighted')
precision_lr = precision_score(y_test, lr_pred, average='weighted')
precision_knn = precision_score(y_test, knn_pred, average='weighted')

# accuracy for each model
accuracy_lr = accuracy_score(y_test, lr_pred)
accuracy_knn = accuracy_score(y_test, knn_pred)
accuracy_dt = accuracy_score(y_test, dt_pred)

# Classification reports for each model
report_lr = classification_report(y_test, lr_pred)
report_dt = classification_report(y_test, dt_pred)
report_knn = classification_report(y_test, knn_pred)

# Printing REsults for Logistic Regression model
print("Logistic Regression Confusion Matrix:")
print(cm_lr)
print("Logistic Regression Recall:", recall_lr)
print("Logistic Regression Precision:", precision_lr)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Classification Report:")
print(report_lr)

# Printing REsults for Decision Tree model
print("\nDecision Tree Confusion Matrix:")
print(cm_dt)
print("Decision Tree Recall:", recall_dt)
print("Decision Tree Precision:", precision_dt)
print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree Classification Report:")
print(report_dt)

# Printing REsults for KNN model
print("\nKNN Confusion Matrix:")
print(cm_knn)
print("KNN Recall:", recall_knn)
print("KNN Precision:", precision_knn)
print("KNN Accuracy:", accuracy_knn)
print("KNN Classification Report:")
print(report_knn)
