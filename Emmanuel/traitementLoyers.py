import json
import pandas as pd

# Charger le fichier JSON
with open("resultats.json", "r") as f:
    data = json.load(f)

# Extraire les `list_id` et `subject` des annonces
ads_data = []
for ad in data["ads"]:
    ads_data.append((ad["list_id"], 
                     ad["first_publication_date"], 
                     ad["price"][0], 
                     ad["location"]["lat"], 
                     ad["location"]["lng"], 
                     ad["url"], 
                     next((attr["value"] for attr in ad["attributes"] if attr["key"] == "rooms"), None),
                     next((attr["value"] for attr in ad["attributes"] if attr["key"] == "square"), None)))


# Créer un DataFrame avec les `list_id` et `subject`
df = pd.DataFrame(ads_data, columns=["list_id", "first_publication_date", "price",
                                     "latitude","longitude", "urls", "nb_pieces","surface"])

# Afficher les premières lignes
# print(df.head())

# Sauvegarder en CSV ou JSON si besoin
# df.to_csv("LoyersFinal.csv", index=False)
data = df
import geopandas as gpd
from pathlib import Path
import folium
from IPython.display import display

gdf = gpd.GeoDataFrame(
    data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

Path("leaflet").mkdir(parents=True, exist_ok=True)
# print("voici gdf")
# print(gdf)
center = gdf[['latitude', 'longitude']].mean().values.tolist()
sw = gdf[['latitude', 'longitude']].min().values.tolist()
ne = gdf[['latitude', 'longitude']].max().values.tolist()

m = folium.Map(location = center, tiles='openstreetmap')

# I can add marker one by one on the map
for i in range(0,len(gdf)):
    folium.Marker([gdf.iloc[i]['latitude'], gdf.iloc[i]['longitude']], popup=gdf.iloc[i]['price']).add_to(m)

m.fit_bounds([sw, ne])

display(m)
# don = pan.DataFrame(gdf)
# df