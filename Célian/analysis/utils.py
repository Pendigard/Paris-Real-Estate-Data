
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import Point
import json

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA


import scipy.stats as stats
from scipy.stats import bartlett, shapiro
import statsmodels.api as sm
import statsmodels.formula.api as smf

import folium
from folium.features import GeoJsonTooltip






def get_periode(annee):
    ref = {1946 : '1946-1970',
        1971 : '1971-1990',
        1990 : 'Apres 1990'}
    periode = 'Avant 1946'
    for annee_ref in ref.keys():
        if annee < annee_ref:
            break
        periode = ref[annee_ref]
    return periode


def increased_rent_from_row(row, df_rent_control):
    mask = (df_rent_control['Numéro du quartier'] == row['quartier_num']) & (df_rent_control['Année'] == 2024)
    verbose = row['quartier']
    if pd.notna(row['building_year']):
        mask &= (df_rent_control['Epoque de construction'] == get_periode(row['building_year']))
        verbose += ' ' + get_periode(row['building_year'])
    if pd.notna(row['nb_rooms']):
        mask &= (df_rent_control['Nombre de pièces principales'] == row['nb_rooms'])
        verbose += ' ' + str(row['nb_rooms'])
    if pd.notna(row['furnished']):
        if row['furnished'] == 2:
            mask &= (df_rent_control['Type de location'] == 'non meublé')
        elif row['furnished'] == 1:
            mask &= (df_rent_control['Type de location'] == 'meublé')
        verbose += ' ' + str(row['furnished'])
    if df_rent_control[mask].empty:
        print(f"Pas de loyer de référence pour {verbose}")
        return np.nan
    return df_rent_control[mask]['Loyers de référence majorés'].mean() if not df_rent_control[mask].empty else np.nan
        

def get_increased_rent(df_rent, rent_control_path='../../data/logement-encadrement-des-loyers.csv'):
    df_rent_control = pd.read_csv(rent_control_path, sep=';')

    df_rent['increased_rent'] = df_rent.apply(lambda row: increased_rent_from_row(row, df_rent_control), axis=1)

    return df_rent


def filter_outliers(df_rent, column='price_m2', quantiles=(0.01, 0.99)):
    df_rent_filtered = df_rent.copy()
    lower_bound = df_rent[column].quantile(quantiles[0])
    upper_bound = df_rent[column].quantile(quantiles[1])
    df_rent_filtered = df_rent_filtered[(df_rent_filtered[column] >= lower_bound) & (df_rent_filtered[column] <= upper_bound)]
    return df_rent_filtered


def map_qualitative_data(df: pd.DataFrame, cols: list, map: dict={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'v': np.nan}) -> pd.DataFrame:
    """
    Mappe les colonnes qualitatives en valeurs numériques.
    :param df: DataFrame à traiter
    :param cols: Liste des colonnes à mapper
    :param map: Dictionnaire de mapping
    :return: DataFrame traité
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(map)
        else:
            print(f"Colonne {col} non trouvée dans le DataFrame.")
    return df

def treat_missing_values(df: pd.DataFrame, columns_to_treat=['surface', 'nb_rooms', 'nb_bedrooms', 'nb_bathrooms', 'floor_number', 'building_year', 'energy_rate'], method='mean', value=None) -> pd.DataFrame:
    """
    Traite les valeurs manquantes dans le DataFrame.
    :param df: DataFrame à traiter
    :param columns_to_treat: Liste des colonnes à traiter
    :param method: Méthode de traitement ('mean', 'median', 'mode')
    :return: DataFrame traité
    """
    df = df.copy()
    if columns_to_treat is None:
        columns_to_treat = df.columns[df.isnull().any()].tolist()
    for col in columns_to_treat:
        if method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0])
        elif method == 'drop':
            df = df.dropna(subset=[col])
        elif method == 'drop_all':
            df= df.dropna()
        elif method == 'value':
            df[col] = df[col].fillna(value)
        else:
            raise ValueError("Méthode non reconnue. Utilisez 'mean', 'median' ou 'mode'.")
    return df

def normalize_data(df: pd.DataFrame, scaler = StandardScaler(), columns_to_normalize=['surface', 'nb_rooms', 'nb_bedrooms', 'nb_bathrooms', 'floor_number', 'building_year', 'energy_rate', 'latitude', 'longitude']) -> pd.DataFrame:
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def threshold_min_lines(df: pd.DataFrame, column: list[str], threshold: int = 10) -> pd.DataFrame:
    """
    Filtre les lignes du DataFrame en fonction d'un seuil minimum d'occurrences dans une colonne donnée.
    :param df: DataFrame à traiter
    :param column: Colonne à filtrer
    :param threshold: Seuil minimum d'occurrences
    :return: DataFrame filtré
    """
    df = df.copy()
    value_to_keep = df[column].value_counts()[df[column].value_counts() >= threshold].index
    df = df[df[column].isin(value_to_keep)]
    return df

# Regression

def fit_regression_model(df: pd.DataFrame, target_column: str = 'price', test_size: float = 0.2, random_state: int = 42, features_column=[], verbose=True) -> tuple:
    X = df[features_column]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    if verbose:
        print(f"\nRMSE sur les données de test : {rmse:.2f}")

    y_pred_train = model.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    if verbose:
        print(f"RMSE sur les données d'entraînement : {rmse_train:.2f}")

    r2 = r2_score(y_test, y_pred)
    if verbose:
        print(f"R² sur les données de test : {r2:.4f}")
        print(f"R² sur les données d'entraînement : {r2_score(y_train, y_pred_train):.4f}")    

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_cv = np.sqrt(-cv_scores)
    if verbose:
        print(f"RMSE avec validation croisée : {rmse_cv.mean():.2f} ± {rmse_cv.std():.2f}")
        print(f"R² avec validation croisée : {cross_val_score(model, X, y, cv=5, scoring='r2').mean():.4f} ± {cross_val_score(model, X, y, cv=5, scoring='r2').std():.4f}")


    coef_df = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    return model, coef_df, rmse, r2

# PCA

def get_pca_components(df: pd.DataFrame, columns: list[str], n_components: int = None) -> tuple:
    pca = PCA(n_components=n_components)
    if columns is not None:
        df = df[columns]
    x_pca = pca.fit_transform(df)
    df_pca = pd.DataFrame(x_pca, columns=[f"PC{i+1}" for i in range(x_pca.shape[1])])
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    return df_pca, cumulative_variance, explained_variance

# Test statistique

def test_bartlett(df, target_col, group_col):
    """
    Teste l'égalité des variances entre les groupes avec le test de Bartlett.
    Hypothèse nulle : toutes les variances sont égales.

    Returns:
        dict: Statistique du test et p-value.
    """
    groups = [group[target_col].values for name, group in df.groupby(group_col)]
    stat, p = bartlett(*groups)
    return {'Bartlett_statistic': stat, 'p_value': p}

def test_normality(df, target_col, group_col):
    results = {}
    for group, data in df.groupby(group_col):
        stat, p = shapiro(data[target_col])
        results[group] = {'W': stat, 'p_value': p}
    return results

def anova_pipeline(df, target_col, group_col):
    df_clean = df[[target_col, group_col]].dropna()

    # Vérification de la normalité
    normality_results = test_normality(df_clean, target_col, group_col)
    normality_violated = False
    for group, result in normality_results.items():
        if result['p_value'] < 0.05:
            print(f"Le groupe {group} ne suit pas une distribution normale (p-value: {result['p_value']:.4f})")
            normality_violated = True
            break

    # Vérification de l'homogénéité des variances
    bartlett_results = test_bartlett(df_clean, target_col, group_col)
    var_homogeneity_violated = False
    if bartlett_results['p_value'] < 0.05:
        print(f"Les variances ne sont pas homogènes (p-value: {bartlett_results['p_value']:.4f})")
        var_homogeneity_violated = True
    else:
        print(f"Les variances sont homogènes (p-value: {bartlett_results['p_value']:.4f})")

    
    groups = [group[target_col].values for name, group in df_clean.groupby(group_col)]

    if normality_violated or var_homogeneity_violated:
        print("ANOVA ne peut pas être appliqué car les conditions ne sont pas remplies.")
        stat, p_val = stats.kruskal(*groups)
        print(f"Test de Kruskal-Wallis : H-statistic = {stat:.4f}, p-value = {p_val:.4f}")
    else:
        stat, p_val = stats.f_oneway(*groups)
        print(f"ANOVA : F-statistic = {stat:.4f}, p-value = {p_val:.4f}")

    result = {
        "F_statistic": stat,
        "p_value": p_val
    }

    return result

# Géometrie
def get_quartier_from_coordinates(lat, lon, gdf):
    """
    Récupère le quartier correspondant à une paire de coordonnées (latitude, longitude).
    :param lat: Latitude
    :param lon: Longitude
    :return: Nom du quartier ou None si pas trouvé
    """
    point = Point(lon, lat)
    point = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(epsg=2154)
    for idx, row in gdf.iterrows():
        if row['Geometry'].contains(point[0]):
            return row['C_QU']
    return None

def get_gdf_quartier(quartier_path='../../data/quartier_paris.csv'):
    """
    Charge le GeoDataFrame des quartiers de Paris.
    :param quartier_path: Chemin vers le fichier CSV des quartiers
    :return: GeoDataFrame des quartiers
    """
    df_quartier = pd.read_csv(quartier_path, sep=';')
    df_quartier['Geometry'] = df_quartier['Geometry'].apply(json.loads)

    # Convertir les geoJson en objets shapely
    df_quartier['Geometry'] = df_quartier['Geometry'].apply(shape)

    gdf_quartier = gpd.GeoDataFrame(df_quartier, geometry='Geometry', crs="EPSG:4326")

    gdf_quartier = gdf_quartier.to_crs(epsg=2154) 

    return gdf_quartier

def get_gdf_quartier_agg(df_rent, quartier_path='../../data/quartier_paris.csv', agg={'price': 'mean', 'surface': 'mean', 'list_id': 'count', 'price_m2': 'mean'}):
    """
    Charge le GeoDataFrame des quartiers de Paris et l'agrège avec les données de loyers.
    :param df_rent: DataFrame des loyers
    :param quartier_path: Chemin vers le fichier CSV des quartiers
    :return: GeoDataFrame agrégé
    """
    gdf_quartier = get_gdf_quartier(quartier_path)
    if 'quartier' not in df_rent.columns:
        df_rent = merge_rent_and_quartier(df_rent, quartier_path)
    
    df_quartier_agg = df_rent.groupby('quartier').agg(agg)

    gdf_quartier_agg = gdf_quartier.merge(df_quartier_agg, left_on='L_QU', right_index=True, how='left')
    gdf_quartier_agg['log_count'] = np.log(gdf_quartier_agg['list_id'])

    return gdf_quartier_agg

def merge_rent_and_quartier(df_rent, quartier_path='../../data/quartier_paris.csv'):
    """
    Fusionne le DataFrame des loyers avec le GeoDataFrame des quartiers.
    :param df_rent: DataFrame des loyers
    :param quartier_path: Chemin vers le fichier CSV des quartiers
    :return: GeoDataFrame fusionné
    """
    gdf_quartier = get_gdf_quartier(quartier_path)
    
    df_rent['quartier_num'] = df_rent.apply(lambda row: get_quartier_from_coordinates(row['latitude'], row['longitude'], gdf_quartier), axis=1)
    df_rent = df_rent.merge(gdf_quartier[['L_QU', 'C_QU']], left_on='quartier_num', right_on='C_QU', how='left')
    df_rent = df_rent.rename(columns={'L_QU': 'quartier'})
    df_rent = df_rent.drop(columns='C_QU')
    return df_rent

def plot_quartier_heatmap(gdf_quartier_agg, column, title, cmap='RdYlGn_r', save_path=None, legend=True, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    gdf_quartier_agg.plot(column=column, ax=ax, legend=legend, cmap=cmap, edgecolor='white', linewidth=0.5, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()

def plot_all_info(gdf_quartier_agg, columns=None, titles=None, cmaps=None):
    if columns is None:
        columns = ['price', 'price_m2', 'surface', 'list_id', 'log_count']
    if titles is None:
        titles = ["Loyer moyen par quartier", "Loyer moyen au m² par quartier", "Surface moyenne par quartier", "Nombre d'annonces par quartier", "Log du nombre d'annonces par quartier"]
    if cmaps is None:
        cmaps = ['RdYlGn_r', 'RdYlGn_r', 'Greens', 'Blues', 'Blues']
    if type(cmaps) == str:
        cmaps = [cmaps] * len(columns)

    for column, title, cmap in zip(columns, titles, cmaps):
        plot_quartier_heatmap(gdf_quartier_agg, column, title, cmap=cmap)


def plot_interactive_quartier_map(gdf_quartier_agg, column, info_columns=None, map_center=(48.8566, 2.3522), zoom_start=12, cmap='YlOrRd'):
    """
    Affiche une carte interactive des quartiers avec info au survol.

    Paramètres :
    - gdf_quartier_agg : GeoDataFrame contenant les géométries des quartiers et les colonnes à afficher.
    - column : nom de la colonne utilisée pour la coloration des quartiers.
    - info_columns : liste des colonnes à afficher au survol (sinon, montre seulement `column`).
    - map_center : tuple (lat, lon) pour centrer la carte.
    - zoom_start : niveau de zoom initial.
    - cmap : palette de couleur (non utilisée directement, mais on peut l'ajouter si besoin).
    """
    
    # On transforme en GeoJSON
    gdf_quartier_agg = gdf_quartier_agg.to_crs(epsg=4326)  # Folium attend du WGS84

    # Création de la carte centrée sur Paris par défaut
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles='CartoDB positron')

    # Préparation des colonnes à afficher dans le tooltip
    if info_columns is None:
        info_columns = [column]

    tooltip = GeoJsonTooltip(
        fields=info_columns,
        aliases=[f"{col}:" for col in info_columns],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 1px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
    )

    # Ajout des polygones à la carte
    folium.GeoJson(
        gdf_quartier_agg,
        tooltip=tooltip,
        style_function=lambda feature: {
            'fillColor': '#3186cc',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6,
        },
    ).add_to(m)

    return m
