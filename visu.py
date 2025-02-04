import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Dans un premier temps on va essayer d'afficher les stats
# sur les logements (perso je travaille sur le fichers des loyers)
# Charger le fichier CSV en DataFrame
loyers = pd.read_csv('data/logement-encadrement-des-loyers.csv', delimiter=';')

Meublement = loyers['Type de location'].value_counts()
FreqMeubl = pd.DataFrame(Meublement)

# On n'apprends rien du tout mdrr
sns.displot(data=FreqMeubl, x="Type de location",weights="count",discrete=True)
plt.show()

# la y'a un peu plus de stats
print(loyers.describe())