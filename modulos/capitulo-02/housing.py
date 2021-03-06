from os import path, makedirs
from six.moves import urllib
import tarfile

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = path.join(path.abspath('.'), 'dados\\housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not path.isdir(housing_path):
        makedirs(housing_path)
    tgz_path = path.join(housing_path, 'housing.tgz')

    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = path.join(housing_path, 'housing.csv')
    return  pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()
housing.info()

housing['ocean_proximity'].value_counts()
housing.describe()

import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize=(20,15))
plt.show()

# Possiveis soluções para separação de dados de treino e teste
import numpy as np

def split_train_test(data: pd.DataFrame, test_ratio: float):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print('%s train - %s test' % (len(train_set), len(test_set)))

import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)

# Transformando um dado continuo em categorico para melhor visualizar 
# a dispersão dos valores em categoria
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

plt.figure(figsize=(8,6))
plt.hist(housing['income_cat'])
plt.ylabel('População')
plt.xlabel('Renda Média')
plt.legend()
plt.title('Categoria de renda')
plt.show()

# Criando uma amostragem estratificada com base na categoria de renda
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Analisando as proporçoes geradas no conjunto de teste
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
housing['income_cat'].value_counts() / len(housing)

# Removendo a coluna income_cat do conjunto de dados
for set_ in (strat_test_set, strat_train_set):
    set_.drop(['income_cat'], axis=1, inplace=True)

# Explorando os dados de treinamento
# fazendo uma copia a fim de não poluir o conjunto de dados
housing = strat_train_set.copy()

# visualizando os dados de latitude e longitude
# c = color
# s = size
plt.figure(figsize=(10,7))
housing.plot(kind='scatter', x = 'longitude', y = 'latitude', alpha=0.4,
 s = housing['population'] / 100, label='população', 
 c = 'median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.title('Latitude x Longitude')
plt.show()

# Buscando a correlação entre os dados
corr_matrix = housing.corr()
corr_matrix.head()

# Visualizando as correlações
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

housing.plot(kind='scatter', x = 'median_income', y='median_house_value', alpha=0.1)

# Combinação de atributos
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedromms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# Preparando para o aprendizado de máquina
housing = strat_train_set.drop(['median_house_value'], axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()

# Limpando os dados
# Existem três opções que podemos usar quando houver instancias com valores missing: 
#   * Livrar-se dos registros 
#   * Livrar-se de todos os atributos
#   * Definir um valor (zero, media, intermediária)

# Podemos fazer isso usando os métodos dropna(), drop() e fillna()
# exemplo:

# housing.dropna(subset=['total_bedrooms']) # opcao 1
# housing.drop('total_bedrooms', axis = 1) # opcao 2
# mediam = housing['total_bedrooms'].median()
# housing['total_bedrooms'].fillna(mediam, inplace=True) # opcao 3

# obs: Se a opção 3, deve calcular o valor médio no conjunto de treinamento e usá-lo para
# preencher os valores faltantes neste, mas não se esqueça de também  salvar o valor médio
# que você calculou. Você precisará dele mais tarde para substituir os valores faltantes no
# conjunto de testes quando quiser avaliar seu sistema e também quando o sistema entrar em
# produção, para substituir os valores faltantes nos novos dados.

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')

# criando uma copia sem variaveis categoricas
housing_num = housing.drop('ocean_proximity', axis = 1)
imputer.fit(housing_num)

# valores das médias de cada variavel continua
imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.info()

# manipulando texto e atributos categoricos
housing_cat = housing['ocean_proximity']
housing_cat_encoded, housing_categories = housing_cat.factorize()

# Onehot encoding
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot.toarray()

encoder.categories_

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y = None):
        return self

    def transform(self, X: np.ndarray, y = None):
        rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
        population_per_household = X[:,population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
attr_adder.transform(housing.values)

# pipelines de transformação
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pepiline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pepiline.fit_transform(housing_num)

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X[self.attributes_names].values

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pepiline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(sparse=False))
])

from sklearn.pipeline import FeatureUnion

full_pepiline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pepiline),
    ('cat_pipeline', cat_pipeline)
])

housing_prepared = full_pepiline.fit_transform(housing)
housing_prepared.shape


# Incluindo nomes das colunas
hdf = pd.DataFrame(housing_prepared)
colunas = num_attribs 
colunas.append("rooms_per_household")
colunas.append("population_per_household")
colunas.append("bedrooms_per_room")

for item in housing_categories:
    if item not in colunas:
        colunas.append(item)

hdf.columns = colunas  
hdf.head()

# Treinando e avaliando o conjunto de treinamento

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# Avaliando com outro modelo
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

# O modelo sofreu super ajuste, ou seja, o modelo decorrou todos os dados
print(tree_rmse)

# Avaliando o modelo com validação cruzada

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
            scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print(tree_rmse_scores)

def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())

display_scores(tree_rmse_scores)

lin_score = cross_val_score(lin_reg, housing_prepared, housing_labels,
                scoring="neg_mean_squared_error", cv = 10)
lin_rmse_score = np.sqrt(-lin_score)
display_scores(lin_rmse_score)

# Testando o modelo com Ensemble Learning
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_prediction = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, forest_prediction)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

forest_score = cross_val_score(forest_reg, housing_prepared, housing_labels, 
                    scoring="neg_mean_squared_error", cv = 10)

forest_rmse_score = np.sqrt(-forest_score)
display_scores(forest_rmse_score)

# Scores geral
display_scores(lin_rmse_score)
print('-'.center(25, '-'))
display_scores(tree_rmse_scores)
print('-'.center(25, '-'))
display_scores(forest_rmse_score)

# Salvando o modelo
from sklearn.externals import joblib

local = path.join(path.abspath('.'), 'modulos\\capitulo-02\\my_model.pkl')
joblib.dump(forest_reg, local)
my_model_loaded = joblib.load(local)

type(my_model_loaded)

# Ajustando o modelo
from sklearn.model_selection import GridSearchCV

param_grid = [
    { 'n_estimators': [3,10,30], 'max_features': [2,4,6,8] },
    { 'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4] }
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

# Avaliando os melhores modelos
feature_importance = grid_search.best_estimator_.feature_importances_
feature_importance

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrroms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_ones_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_ones_hot_attribs
sorted(zip(feature_importance, attributes), reverse=True)

# Avaliando o sistema no conjunto de testes
final_model = grid_search.best_estimator_
final_model

X_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pepiline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
print(final_mse)

