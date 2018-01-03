import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])
df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}

df['size'] = df['size'].map(size_mapping)

# X = df[['color', 'size', 'price']].values

# color_le = LabelEncoder()
# X[:, 0] = color_le.fit_transform(X[:, 0])
# ohe = OneHotEncoder(categorical_features=[0], sparse=False)
# print(ohe.fit_transform(X))


class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
df_y = pd.DataFrame({ 'class': y })

dummy_features = pd.get_dummies(df[['price', 'color', 'size']])

df = pd.concat([dummy_features, df_y], axis=1)
print(df)
