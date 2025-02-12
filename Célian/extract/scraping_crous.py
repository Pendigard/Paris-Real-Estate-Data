#%%

# IMPORTS

import bs4
from lxml import etree
import requests
import pandas as pd
import re
import sys
import csv
from adress_to_coord import adress_to_coord
#%%

# GET THE LINKS OF THE RESIDENCES

link = 'https://www.crous-paris.fr/se-loger/liste-de-nos-logements/'

response = requests.get(link, verify=False)
soup = bs4.BeautifulSoup(response.text, 'html.parser')

dom = etree.HTML(str(soup))
residence_links = dom.xpath('//a[contains(@href, "https://www.crous-paris.fr/logement/")]/@href')

# %%

# STORE THE PAGES OF THE RESIDENCES IN A CSV FILE
# Le crous bloque les requêtes après trop de demandes. Il n'y a pas beaucoup de résidences donc on 
# peut stocker les pages dans un fichier csv pour les traiter plus tard

residences = []
residences_pages = []
for link in residence_links:
    response = requests.get(link, verify=False)
    dico = {}
    dico['link'] = link
    dico['page'] = response.text
    residences_pages.append(dico)

df_pages = pd.DataFrame(residences_pages)
df_pages.to_csv('../../data/crous_pages.csv', index=False, sep='¥')
# %%

# LOAD THE CSV FILE

csv.field_size_limit(int(sys.maxsize/10))

df_pages = pd.read_csv('../../data/crous_pages.csv', sep='¥')
# %%

# EXTRACT INFORMATION FROM THE PAGES


def extract_information(loc_content, page_link):
    apparts =[]
    pattern = re.compile(r"\b\d{5}\b")
    match = pattern.search(loc_content[0] + " " + loc_content[1])
    code_postal = ''
    if match:
        code_postal = match.group()

    address = loc_content[0].split('\xa0')[0]
    pattern = re.compile(r"T([1-9]).*?(\d{1,4})\s*m².*?(\d{1,4})\s*€")
    # Pattern pour reconnaître des phrase de type "TX ... Y m² ... Z €"
    for i in range(1, len(loc_content)):
        appart = {}
        if '€' in loc_content[i]:
            match = pattern.search(loc_content[i])
            if match:
                appart['adresse'] = address
                appart['code_postal'] = code_postal
                lon, lat = adress_to_coord(address + ' ' + code_postal)
                appart['lat'] = lat
                appart['lon'] = lon
                appart['taille'] = f'T{match.group(1)}'
                appart['surface'] = int(match.group(2))
                appart['loyer'] = int(match.group(3))
                appart['_src'] = page_link
                apparts.append(appart)
            if not match:
                print(loc_content[i])
    return apparts
    
df = []
for i in range(len(df_pages)):
    soup = bs4.BeautifulSoup(df_pages['page'][i], 'html.parser')
    dom = etree.HTML(str(soup))
    page_link = df_pages['link'][i]
    loc_content = dom.xpath('//div[@class="localisation_content"]/ul/li/text()')
    df += extract_information(loc_content, page_link)


    # soup = bs4.BeautifulSoup(response.text, 'html.parser')
    # dom = etree.HTML(str(soup))
    # loc_content = dom.xpath('//div[@class="localisation_content"]/ul/li/text()')
    # extract_information(loc_content)

# %%

# TO CSV

# Le crous représente 7550 logements à Paris environ
# avec un taux d'occupation moyen de 95%
# Il y aurait donc 378 logements libres


df = pd.DataFrame(df)
df.to_csv('../../data/crous_appart2.csv')


# %%


df = pd.read_csv('../../data/crous_appart.csv', sep=',')

# %%
