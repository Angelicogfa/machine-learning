import numpy as np

def sort_by_target(mnist):
    reorder_train = np.array(sorted([target, i] for i, target in enumerate(mnist.target[:6000])))[:, 1]
    reorder_test = np.array(sorted([target, i] for i, target in enumerate(mnist.target[6000:])))[:, 1]
    mnist.data[:6000] = mnist.data[reorder_train]
    mnist.target[:6000] = mnist.target[reorder_train]
    mnist.data[6000:] = mnist.data[reorder_test + 6000]
    mnist.target[6000:] = mnist.target[reorder_test + 6000]

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784',  version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)
    sort_by_target(mnist)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')

x, y = mnist['data'], mnist['target']

x.shape
y.shape

from matplotlib.cm import binary
import matplotlib.pyplot as plt

some_digit = x[3550]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = binary, interpolation="nearest")
plt.axis('off')
plt.show()

y[3550]

## Classificador de números 5

X_train, X_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgdc_class = SGDClassifier(random_state=42)
sgdc_class.fit(X_train, y_train_5)

sgdc_class.predict([some_digit])

## Validação cruzada

# Implementando a validação manualmente
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgdc_class)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_corret = sum(y_pred == y_test_folds)
    print(n_corret/len(y_pred))

# Implementando a validação utilizando a função cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score(sgdc_class, X_train, y_train_5, cv=3, scoring="accuracy")

## Matriz de confusão

# Treinando o modulo com validação cruzada
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgdc_class, X_train, y_train_5, cv=3)

# Gerando a matriz de confusão
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
confusion_matrix(y_train_5, y_train_pred)

# [TN FP] 
# [FN TP]

precision_score(y_train_5, y_train_pred) # TP / (TP + FP)
recall_score(y_train_5, y_train_pred) # TP / (TP + FN)
f1_score(y_train_5, y_train_pred) # TP / (TP + (FN + FP) / 2)

y_score = sgdc_class.decision_function([some_digit])
threshold = 0
y_some_digit_pred = (y_score > threshold)
y_some_digit_pred

threshold = 200000
y_some_digit_pred = (y_score > threshold)
y_some_digit_pred

y_score = cross_val_predict(sgdc_class, X_train, y_train_5, cv=3, method='decision_function')

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_score)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label = 'Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='center left')
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

y_train_pred_90 = (y_score > 70000)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

# Curva ROC
from sklearn.metrics import roc_curve
help(roc_curve)

fpr, tpr, thresholds = roc_curve(y_train_5, y_score)

def plot_roc_curve(fpr,tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.plot([0,1,0,1])
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_score)

# Validando com um RamdonForestClassifier
from sklearn.ensemble import  RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3, method='predict_proba')

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc='lower right')
plt.show()

roc_auc_score(y_train_5, y_scores_forest)