######################################################################################################################
# Visualizing data:

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data_train = pd.read_csv('train.csv')


def label_encoder(data):
    for f in data.columns:
        if data[f].dtypes == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
    return data


data_train = label_encoder(data_train)

print(data_train.info())
print(data_train.head())


def my_plotter(df, save_path):
    for col in df.columns:
        sns_plot = sns.jointplot(x=col, y='target', data=data_train, dropna=True)
        sns_plot.savefig(save_path + col + '_relation_ship_with_response.png')
    return 1


path = '/home/pbadmin/Downloads/Kaggle_BNP/Visualization/'

my_plotter(data_train, path)
