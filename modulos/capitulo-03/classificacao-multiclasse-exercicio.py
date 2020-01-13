from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.transform import rotate
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
from os import path
from sklearn.datasets import load_digits

data, target = load_digits(10, True)

# Config
np.random.seed(42)

# train test
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=42)

# -----------------------------------------------------------------------------------

# 1 - Classificador com 97% precisao ou mais


def print_confusion_matrix(pred):
    return pd.DataFrame(pred, columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])


# Dummy Classifier para obter uma acuracia minima
dummy = DummyClassifier()
dummy.fit(x_train, y_train)

dummy_predict = dummy.predict(x_train)

accuracy_score(y_train, dummy_predict)
confusion_matrix(y_train, dummy_predict)

# Multilabel classifier
model = OneVsRestClassifier(DecisionTreeClassifier())
model.fit(x_train, y_train)

y_train_predict = model.predict(x_train)
accuracy_score(y_train, y_train_predict)
confusion_matrix(y_train, y_train_predict)

y_test_predict = model.predict(x_test)
accuracy_score(y_test, y_test_predict)

print_confusion_matrix(confusion_matrix(y_test, y_test_predict))

# Decision tree
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_train_predict = model.predict(x_train)
accuracy_score(y_train, y_train_predict)

y_test_predict = model.predict(x_test)
accuracy_score(y_test, y_test_predict)

# Modelo
params = {"n_neighbors": [3, 5, 8, 13], "weights": [
    "uniform", "distance"], "metric": ["euclidean", "manhattan"]}

g_search = GridSearchCV(KNeighborsClassifier(), params,
                        verbose=1, cv=5,  n_jobs=-1)
g_search.fit(x_train, y_train)

g_search.best_params_
g_search.best_score_
model = g_search.best_estimator_

y_model_predict = model.predict(x_test)
accuracy_score(y_test, y_model_predict)
print_confusion_matrix(confusion_matrix(y_test, y_model_predict))

# escalando os dados

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
g_search.fit(x_train_scaled, y_train)

g_search.best_params_
g_search.best_score_
model = g_search.best_estimator_

y_model_predict = model.predict(scaler.transform(x_test))
accuracy_score(y_test, y_model_predict)
print_confusion_matrix(confusion_matrix(y_test, y_model_predict))

# --------------------------------------------------------------------------------------

# 2- Expandir o conjunto de dados incluindo imagens rotacionadas aumentando assim o
# conjunto de possibilidades conhecidas para o modelo
# Essa técnica é denominada 'Data Augmentation'

# Usando a biblioteca scikit-image, para instalar o pacote rodar o comando abaixo:
#  !pip install scikit-image


def rotate_image(img_array, degree):
    image = img_array.reshape(8, 8)
    return rotate(image, degree).reshape(64)


def print_image(img_array, color='Greys'):
    image = img_array.reshape(8, 8)
    plt.imshow(image, cmap=color)


x_train_augmented = [x for x in x_train]
y_train_augmented = [y for y in y_train]

for position in [90, 180, 270]:
    for image, label in zip(x_train, y_train):
        x_train_augmented.append(rotate_image(image, position))
        y_train_augmented.append(label)

x_train_augmented = np.array(x_train_augmented)
y_train_augmented = np.array(y_train_augmented)

knn_class = KNeighborsClassifier(**g_search.best_params_)
knn_class.fit(x_train_augmented, y_train_augmented)

y_train_augmented_predict = knn_class.predict(x_train_augmented)
accuracy_score(y_train_augmented, y_train_augmented_predict)

y_test_augmented_predict = knn_class.predict(x_test)
accuracy_score(y_test, y_test_augmented_predict)

# -----------------------------------------------------------------------------
# 3 - Encarrar o conjunto de dados do titanic

np.random.seed(42)

arquivos = path.join(path.abspath('..\..\\'), "dados")

data = pd.read_csv(path.join(arquivos, "titanic\\train.csv"))

features = np.array(["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
target = np.array(["Survived"])

columns = np.append(features, target)


class SelectFields(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


class ReplaceValue(BaseEstimator, TransformerMixin):
    def __init__(self, column, replace_map, fix_type=None):
        self.column = column
        self.replace_map = replace_map
        self.fix_type = fix_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.column] = X[self.column].map(self.replace_map)

        if self.fix_type != None:
            X[self.column] = X[self.column].astype(self.fix_type)

        return X



class ToDataset(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.columns)

data["Embarked"].str.get_dummies()
data["Sex"].str.get_dummies()


cat_pipeline = Pipeline([
    ("select_fields", SelectFields(columns=["Pclass", "Sex", "Embarked"])),
    ("most_frequent_imputer", SimpleImputer(strategy="most_frequent")),
    ("encoding", OneHotEncoder(sparse=False))
])

cat_pipeline.fit_transform(data)

num_pipeline = Pipeline([
    ("select_fields", SelectFields(columns=["Age", "SibSp", "Parch", "Fare"])),
    ("median_imputer", SimpleImputer(strategy="median"))
])

preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

x = preprocess_pipeline.fit_transform(data[features])
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y)


# Tree classifier
tree_class = DecisionTreeClassifier()
tree_class.fit(x_train, y_train)

predict_train = tree_class.predict(x_train)
accuracy_score(y_train, predict_train)
confusion_matrix(y_train, predict_train)

predict_test = tree_class.predict(x_test)
tree_accuracy = accuracy_score(y_test, predict_test)
confusion_matrix(y_test, predict_test)

# SVM
svc = SVC()
svc.fit(x_train, y_train)

predict_train = svc.predict(x_train)
accuracy_score(y_train, predict_train)
confusion_matrix(y_train, predict_train)

predict_test = svc.predict(x_test)
svm_accuracy = accuracy_score(y_test, predict_test)
confusion_matrix(y_test, predict_test)

# Gaussean
gaussian = GaussianProcessClassifier()

gaussian.fit(x_train, y_train)
train_predict = gaussian.predict(x_train)
accuracy_score(y_train, train_predict)
confusion_matrix(y_train, train_predict)

test_predict = gaussian.predict(x_test)
gaussean_accuracy = accuracy_score(y_test, test_predict)
confusion_matrix(y_test, test_predict)

# Naive Bayes
nb = GaussianNB()

nb.fit(x_train, y_train)
train_predict = nb.predict(x_train)
accuracy_score(y_train, train_predict)
confusion_matrix(y_train, train_predict)

test_predict = nb.predict(x_test)
nv_accuracy = accuracy_score(y_test, test_predict)
confusion_matrix(y_test, test_predict)

# RandomFlorest
from sklearn.ensemble import RandomForestClassifier
rn_forest = RandomForestClassifier()
rn_forest.fit(x_train, y_train)

train_predict = rn_forest.predict(x_train)
accuracy_score(y_train, train_predict)
confusion_matrix(y_train, train_predict)

test_predict = rn_forest.predict(x_test)
rf_accuracy = accuracy_score(y_test, test_predict)
confusion_matrix(y_test, test_predict)

# plot de comparação
plt.bar(["tree", "svm", "gaussean", "nb", "random forest"], [tree_accuracy, svm_accuracy, gaussean_accuracy, nv_accuracy, rf_accuracy])
plt.xlabel("Modelos")
plt.ylabel("Accuracy")
plt.title("Accuracy X modelo")
plt.show()