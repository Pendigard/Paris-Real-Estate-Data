import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def filter_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['surface'].notna()) & (df['furnished'].notna())].copy()

    for col in ['nb_rooms', 'nb_bedrooms', 'nb_bathrooms', 'floor_number', 'building_year']:
        df[col] = df[col].fillna(df[col].mean())

    df['elevator'] = df['elevator'].replace({2: 0, 0: 1})
    df['elevator'] = df['elevator'].fillna(df['elevator'].mean())

    df['parking'] = df['parking'].fillna(0)

    energy_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'v': np.nan}
    df['energy_rate'] = df['energy_rate'].str.lower().map(energy_mapping)
    df['energy_rate'] = df['energy_rate'].fillna(df['energy_rate'].mean())

    specific_cols = [col for col in df.columns if col.startswith('specific_type') or col.startswith('specificities') or col.startswith('outside_access')]
    specific_cols.remove('specific_type_Autres')
    for col in specific_cols:
        df[col] = df[col].map({'False': 0, '1.0': 1})
        df[col] = df[col].astype(float)


    return df[['furnished', 'surface', 'nb_rooms', 'nb_bedrooms', 'nb_bathrooms', 'floor_number', 'building_year', 'elevator', 'parking', 'energy_rate'] + specific_cols]

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    columns_to_normalize = ['surface', 'nb_rooms', 'nb_bedrooms', 'nb_bathrooms', 'floor_number', 'building_year', 'energy_rate']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df




