#%%
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# Charger le fichier JSON
with open("../../Emmanuel/resultats_2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraire les `list_id` et `subject` des annonces
ads_data = []
for ad in data["ads"]:
    ads_data.append({
        "list_id": ad["list_id"],
        "type": next((attr["value_label"] for attr in ad["attributes"] if attr["key"] == "real_estate_type"), None),
        "specific_type": next((attr["values_label"] for attr in ad["attributes"] if attr["key"] == "real_estate_type_specificities"), None),
        "first_publication_date": ad["first_publication_date"],
        "zipcode": ad["location"]["zipcode"],
        "district_id": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "district_id"), None),
        "price": ad["price"][0],
        "charges_included": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "charges_included"), None),
        "furnished": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "furnished"), None),
        "surface": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "square"), None),
        "nb_rooms": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "rooms"), None),
        "nb_bedrooms": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "bedrooms"), None),
        "nb_bathrooms": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "nb_bathrooms"), None),
        "specificities": next((attr["values_label"] for attr in ad["attributes"] if attr["key"] == "specificities"), None),
        "floor_number": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "floor_number"), None),
        "floor_building": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "nb_floors_building"), None),
        "outside_access": next((attr["values_label"] for attr in ad["attributes"] if attr["key"] == "outside_access"), None),
        "elevator": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "elevator"), None),
        "parking": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "nb_parkings"), None),
        "building_year": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "building_year"), None),
        "security_deposit": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "security_deposit"), None),
        "energy_rate" : next((attr["value"] for attr in ad["attributes"] if attr["key"] == "energy_rate"), None),
        "greenhouse_rate" : next((attr["value"] for attr in ad["attributes"] if attr["key"] == "ges"), None),
        "owner_type" : ad["owner"]["type"],
        "owner_company" : ad["owner"]["name"] if ad["owner"]["type"] == "pro" else None,
        "latitude": ad["location"]["lat"],
        "longitude": ad["location"]["lng"],
        "url": ad["url"]
    })

# Créer un DataFrame avec les `list_id` et `subject`
df = pd.DataFrame(ads_data)

df['price'] = df['price'].astype(float)
df['surface'] = df['surface'].astype(float)
df['price_m2'] = df['price'] / df['surface']

# Réorganiser les colonnes

df = df[
    [
        "list_id",
        "type",
        "specific_type",
        "first_publication_date",
        "zipcode",
        "district_id",
        "price",
        "charges_included",
        "furnished",
        "surface",
        "price_m2",
        "nb_rooms",
        "nb_bedrooms",
        "nb_bathrooms",
        "specificities",
        "floor_number",
        "floor_building",
        "outside_access",
        "elevator",
        "parking",
        "building_year",
        "security_deposit",
        "energy_rate" ,
        "greenhouse_rate" ,
        "owner_type" ,
        "owner_company" ,
        "latitude",
        "longitude",
        "url"
    ]
]

def explode_list(df, col):
    # 1. Nettoyer les NaN → listes vides
    df[f'{col}_clean'] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    # 2. Exploser la colonne
    df_exploded = df[[f'{col}_clean']].explode(f'{col}_clean').copy()
    df_exploded['value'] = True

    # 3. Pivot table → One-hot encoding
    df_one_hot = df_exploded.pivot_table(
        index=df_exploded.index,
        columns=f'{col}_clean',
        values='value',
        fill_value=False
    )

    # 4. Renommer les colonnes binaires pour éviter collisions
    df_one_hot.columns = [f"{col}_{str(c)}" for c in df_one_hot.columns]

    # 5. Nettoyage de la colonne temporaire et fusion
    df.drop(columns=[f'{col}_clean'], inplace=True)
    return df.join(df_one_hot)

# --- Application sur le DataFrame ---

# Garde les anciennes colonnes pour ne pas faire fillna dessus
original_columns = df.columns.tolist()

# Applique à chaque colonne cible
for col in ['specific_type', 'specificities', 'outside_access']:
    df = explode_list(df, col)

# Supprime les colonnes d’origine (avec les listes)
df.drop(columns=['specific_type', 'specificities', 'outside_access'], inplace=True)

# Ne remplir de 0 que les nouvelles colonnes binaires (pas celles d'origine)
new_columns = [col for col in df.columns if col not in original_columns]
df[new_columns] = df[new_columns].fillna(False)

# Export CSV
df.to_csv("../../data/rent_2.csv", index=False, encoding="utf-8", sep=";", header=True)
# %%
