# 
#
# %%
import requests

link = 'https://api-adresse.data.gouv.fr/search/?'


def adress_to_link(adress):
    return link + 'q=' + adress.replace(' ', '+')

def adress_to_coord(adress):
    link = adress_to_link(adress)
    response = requests.get(link)
    return response.json()['features'][0]['geometry']['coordinates']

if __name__ == '__main__':
    addr = adress_to_coord('1 rue de la liberté 35000')


# %%
