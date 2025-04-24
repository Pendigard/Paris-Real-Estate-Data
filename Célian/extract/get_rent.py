#%%
import json
import pandas as pd

# Charger le fichier JSON
with open("../../Emmanuel/resultats.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraire les `list_id` et `subject` des annonces
ads_data = []
for ad in data["ads"]:
    ads_data.append({
        "list_id": ad["list_id"],
        "type": next((attr["value_label"] for attr in ad["attributes"] if attr["key"] == "real_estate_type"), None),
        "specific_type": next((attr["value_label"] for attr in ad["attributes"] if attr["key"] == "real_estate_type_specificities"), None),
        "first_publication_date": ad["first_publication_date"],
        "zipcode": ad["location"]["zipcode"],
        "district_id": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "district_id"), None),
        "price": ad["price"][0],
        "charges_included": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "charges_included"), None),
        "furnished": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "furnished"), None),
        "surface": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "square"), None),
        "nb_pieces": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "rooms"), None),
        "nb_chambres": next((attr["value"] for attr in ad["attributes"] if attr["key"] == "bedrooms"), None),
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

df.to_csv("../../data/rent.csv", index=False, encoding="utf-8", sep=";", header=True)
# %%
