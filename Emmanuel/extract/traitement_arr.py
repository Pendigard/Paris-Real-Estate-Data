import urllib.request as request
import bs4
import pandas as pd
from time import sleep

def extract_notes(url, arrondissement_name):
    sleep(10)  
    try:
        request_text = request.urlopen(url).read()
        page = bs4.BeautifulSoup(request_text, "lxml")
        table = page.find('table', {'id': 'tablonotes'})
        tr = table.find_all('tr')

        notes = {}
        for row in tr:
            th = row.find('th')
            td = row.find('td')
            if th and td:
                critere = th.text.strip()
                note = td.text.strip().replace(',', '.')  # pour convertir en float ensuite
                notes[critere] = float(note)
        return arrondissement_name, notes
    except Exception as e:
        print(f"Erreur pour {arrondissement_name}: {e}")
        return arrondissement_name, {}

# Génération des URLs pour les 20 arrondissements
base_url = "https://www.ville-ideale.fr/paris-{}e-arrondissement_751{:02d}"
arr_data = {}

# for i in range(16, 21):
#     if i == 1:
#         arr_name = f"Paris {i}er"
#     else:
#         arr_name = f"Paris {i}e"

#     url = base_url.format(i, i)
#     name, notes = extract_notes(url, arr_name)
#     arr_data[name] = notes

# Transformer en DataFrame
# df = pd.DataFrame.from_dict(arr_data, orient='columns')
# df.to_csv('/Users/emmanuel/Documents/M1/S2/DALAS/Projet/Paris-Real-Estate-Data/Emmanuel/extract/arrondissements_notes_tmp.csv', index=True)



# Lance pour visu la df dans l'interpreteur interactif
# ce sont des notes / 10
data = pd.read_csv('../../data/arrondissements_notes.csv',delimiter=',')
data


