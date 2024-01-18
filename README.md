# Nom du Projet


**FootChaos** est un jeu en 2D reprenant les principes de RocketLeague intégrant une
intelligence artificielle avancée, développé spécifiquement pour les puces
d'Apple (M1, M2, ...) en utilisant **metal**

## Captures d'écran

()

## Avant - Après

| Nombre de réseau | Nombre de couche | Taille des groupes | Taille des couche | Temps CPU | Temps GPU | Gain |
|------------------|------------------|--------------------|-------------------|-----------|-----------| ---- |
| 500              | 50               | 10                 | 15                | 144s      | 35s       | 75%  |
| 1000             | 10               | 10                 | 15                | 50s       | 43s       | 14%  |
| 1000             | 20               | 10                 | 15                | 115s      | 71s       | 39%  |
| 1000             | 20               | 50                 | 15                | 600s      | 669s      | -11% |
| 2000             | 50               | 20                 | 15                | 1259s     | 1494s     | -19% |
|                  |                  |                    |                   |           |           |      |
