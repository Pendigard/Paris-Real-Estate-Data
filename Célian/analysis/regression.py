import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

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

    df = pd.get_dummies(df, columns=['zipcode'], drop_first=True, prefix='zipcode')
    zipcode_cols = [col for col in df.columns if col.startswith('zipcode_')]
    return df[['price', 'price_m2', 'furnished', 'surface', 'nb_rooms', 'nb_bedrooms', 'nb_bathrooms', 'floor_number', 'building_year', 'elevator', 'parking', 'energy_rate', 'latitude', 'longitude'] + specific_cols + zipcode_cols]

def normalize_data(df: pd.DataFrame, columns_to_normalize=['surface', 'nb_rooms', 'nb_bedrooms', 'nb_bathrooms', 'floor_number', 'building_year', 'energy_rate', 'latitude', 'longitude']) -> pd.DataFrame:
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def fit_regression_model(df: pd.DataFrame, target_column: str = 'price', test_size: float = 0.2, random_state: int = 42, features_column=[]) -> tuple:
    X = df[features_column]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"\nRMSE sur les données de test : {rmse:.2f}\n")

    y_pred_train = model.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    print(f"RMSE sur les données d'entraînement : {rmse_train:.2f}\n")

    coef_df = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print("Coefficients les plus influents :")
    print(coef_df.to_string(index=False))

    return model, coef_df