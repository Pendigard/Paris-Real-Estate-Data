# BrainStorm

## Problematique

### Pourquoi c'est difficile de se loger à Paris pour un étudiant?

## Idées

- faire le prix/m2 **[Fait]**
- normaliser **[Fait]**
- a quel point les proprios sont frauduleux en fonction des quartiers
- qualité de vie du quartier (en fonction du prix)
- voir l'encadrement des loyers

- Par le prof JNV 26/03/2035
  - recupérer les tendances d'annonces (par saison) par mois par exemples
  - relation entre les lieux d'etudes et les logements (ça se vent bien ?)
    - voir les corrélations si c'est une causalité
  - voir les stats sur les quariters (criminalité) ? influence sur le prix ?

  - joindre les tables
  - inventer une tache de prédiction
  - partir d'une hypothèse et la prouver avec les données trouvés

## représentations intéressantes [les attendus]

- Eviter de traiter les tables de manière disparate

## Problématiques

1. Où est le campus? **[vu]**
2. Combien je peux dépenser pour un logement? **[vu]**
3. Quelle est le poids des aides au logement? **[skip]**
4. Comment évolue le loyer? **[vu]**
5. Y'a-t-il suffisament de logements pour les étudiants?
6. Quels sont les alternatives à la location classique? (crous) **[vu]**
7. Quels sont les critères de selection? **[skip]**
8. Quels quartiers sont les plus adaptés?
9. Quels sont les moyens de transport?
10. qualité du logement commerces de proximité, lignes de transports, super-marché, laverie
11. trouver logement en fonction du profil de l'etudiant (POIs)

## Sources

1. [Seloger(4,5,7)](https://www.seloger.com/)

2. [Valeur Foncières(4)](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/#/resources)

3. [Établissement(1)](https://data.enseignementsup-recherche.gouv.fr/explore/dataset/fr-esr-principaux-etablissements-enseignement-superieur/export/?disjunctive.type_d_etablissement&disjunctive.typologie_d_universites_et_assimiles&sort=uo_lib)

4. [ObservatoireDesLoyers(4)](https://www.observatoire-des-loyers.fr/donnees-annee) (info sur les loyers)

5. [VilleDeParis(4)](https://opendata.paris.fr/explore/dataset/logement-encadrement-des-loyers/export/?disjunctive.nom_quartier&disjunctive.piece&disjunctive.epoque&disjunctive.meuble_txt&disjunctive.id_zone&disjunctive.annee) (info sur les loyers)

6. [Données gouvernementales](https://www.data.gouv.fr/fr/) (polyvalant)

## TODO

Célian -> donnes de sélectivité de logements, enadrement des loyers, merge les tables, turn over

Emmanuel -> différentes stats sur chaque quartiers  

## Utilisation de modèle

Le modele retourne un vecteur qui represente un score par appratement en fonction du profil d'un etudiant  
il retourne un vecteur d'appartement

## Indications sur le projet

- présenter le projet
- décrire le dataset
- l'objectif des analyses
- présenter les analyses correspondants aux problématiques
- présentation des résultats des graphes
- commenter les graphes  
- dans la conlusion donner la reponse a la problématique
- les difficultés rencontrés dans le projet
- l'implication de chaque participant
- penser à vulgariser les résultats (cours sur le storytelling)

## Rendu du projet

Dans un fichier zip mettre  

- Rendu technique (pdf)
- code + dataset
- Slides de la présentation
- Video de 10 minutes (vraiment faire 10 minutes)

A la fin il y'aura un oral de 10 à 15 minutes de questions (en mai à préciser)
