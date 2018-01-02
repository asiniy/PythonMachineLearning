import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

df['classlabel'] = y

print(df)
