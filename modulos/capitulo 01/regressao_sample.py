import matplotlib
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

arquivos = path.join(path.abspath('.'), "dados")
oecd_bli = pd.read_csv(path.join(arquivos, "oecd_bli_2015.csv"), thousands=',')
gdp_per_capita = pd.read_csv(path.join(arquivos, "gdp_per_capita.csv"), thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

oecd_bli.head()
gdp_per_capita.head()

def prepare_country_stats(oecd: pd.DataFrame, gdp: pd.DataFrame):
    db_join = gdp_per_capita.merge(oecd_bli, on='Country')
    print(db_join.head())
    db_join.columns = ['Country', 'GDP per capita', 'Life satisfaction']
    # db_join['GDP per capita'] = np.round(db_join['GDP per capita'], 0)
    db_join = db_join.dropna()
    return db_join

oecd_bli = oecd_bli[(oecd_bli['Unit']=='Average score') & \
        (oecd_bli['INEQUALITY']=='TOT') & \
        (oecd_bli['Indicator']=='Life satisfaction')][['Country', 'Value']]

gdp_per_capita = gdp_per_capita[gdp_per_capita['Estimates Start After']==2015.0][['Country', '2015']]

oecd_bli.head()
gdp_per_capita.head()

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

country_stats.head(10)

X = np.array(country_stats['GDP per capita']).reshape(-1,1)
Y = np.array(country_stats['Life satisfaction']).reshape(-1,1)

plt.scatter(X, Y)
plt.xlabel('GDP per capita')
plt.ylabel('Life satisfaction')
plt.title('GDP per capita x Life satisfaction')
plt.show()

model = LinearRegression()
model.fit(X, Y)

x_new = [[22587.0]]
print('Score: %s' % (model.predict(x_new)))


