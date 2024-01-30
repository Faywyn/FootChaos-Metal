import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import sys
import os

idTraining = sys.argv[1]

# Charger les données initiales depuis le CSV
path = (os.getcwd() + "/trainings/" + idTraining + "/metrics.csv")
data = pd.read_csv(path, header=None, names=['Generation', 'Best', 'Average', 'Time'], sep=";")

# Initialiser le graphique avec trois sous-graphiques côte à côte
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
line1, = ax1.plot(data['Generation'], data['Best'], label='Best', linestyle='-', linewidth=0.5)
line2, = ax2.plot(data['Generation'], data['Average'], label='Average', linestyle='-', linewidth=0.5)
line3, = ax3.plot(data['Generation'], data['Time'] / 1000, label='Time', linestyle='-', linewidth=0.5)

# Ajouter un titre à chaque sous-graphique
ax1.set_title('Best')
ax2.set_title('Average')
ax3.set_title('Time')

# Calculer la moyenne mobile avec une fenêtre de 5 points
nbPoint = int(sys.argv[2])
rolling_avg1 = data['Best'].rolling(window=nbPoint).mean()
rolling_avg2 = data['Average'].rolling(window=nbPoint).mean()
rolling_avg3 = (data['Time'] / 1000).rolling(window=nbPoint).mean()

# Ajouter les lignes de moyenne mobile avec une couleur différente et une légende
ax1.plot(data['Generation'], rolling_avg1, label='Moving Average', color='red', linestyle='-', linewidth=1.5)
ax2.plot(data['Generation'], rolling_avg2, label='Moving Average', color='red', linestyle='-', linewidth=1.5)
ax3.plot(data['Generation'], rolling_avg3, label='Moving Average', color='red', linestyle='-', linewidth=1.5)

# Ajouter la légende de la moyenne mobile à chaque sous-graphique
ax1.legend()
ax2.legend()
ax3.legend()

# Ajouter une ligne de tendance à chaque sous-graphique
def add_trendline(ax, x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), label='Trend', color='green', linestyle='-', linewidth=1.5)

add_trendline(ax1, data['Generation'], data['Best'])
add_trendline(ax2, data['Generation'], data['Average'])
add_trendline(ax3, data['Generation'], data['Time'] / 1000)

# Ajouter la légende de la tendance à chaque sous-graphique
ax1.legend()
ax2.legend()
ax3.legend()

# Fonction d'animation appelée à chaque mise à jour
def update(frame):
    # Charger les nouvelles données depuis le CSV avec le séparateur ;
    new_data = pd.read_csv(path, header=None, names=['Generation', 'Best', 'Average', 'Time'], sep=';')

    # Mettre à jour les données des graphiques
    line1.set_xdata(new_data['Generation'])
    line1.set_ydata(new_data['Best'])

    line2.set_xdata(new_data['Generation'])
    line2.set_ydata(new_data['Average'])

    line3.set_xdata(new_data['Generation'])
    line3.set_ydata(new_data['Time'] / 1000)

    # Calculer la nouvelle moyenne mobile avec une fenêtre de 5 points
    new_rolling_avg1 = new_data['Best'].rolling(window=nbPoint).mean()
    new_rolling_avg2 = new_data['Average'].rolling(window=nbPoint).mean()
    new_rolling_avg3 = (new_data['Time'] / 1000).rolling(window=nbPoint).mean()

    # Mettre à jour les données de la ligne de moyenne mobile
    ax1.lines[1].set_xdata(new_data['Generation'])
    ax1.lines[1].set_ydata(new_rolling_avg1)

    ax2.lines[1].set_xdata(new_data['Generation'])
    ax2.lines[1].set_ydata(new_rolling_avg2)

    ax3.lines[1].set_xdata(new_data['Generation'])
    ax3.lines[1].set_ydata(new_rolling_avg3)

    # Mettre à jour les données de la ligne de tendance
    ax1.lines[2].set_xdata(new_data['Generation'])
    ax1.lines[2].set_ydata(np.poly1d(np.polyfit(new_data['Generation'], new_data['Best'], 1))(new_data['Generation']))

    ax2.lines[2].set_xdata(new_data['Generation'])
    ax2.lines[2].set_ydata(np.poly1d(np.polyfit(new_data['Generation'], new_data['Average'], 1))(new_data['Generation']))

    ax3.lines[2].set_xdata(new_data['Generation'])
    ax3.lines[2].set_ydata(np.poly1d(np.polyfit(new_data['Generation'], new_data['Time'] / 1000, 1))(new_data['Generation']))

    # Ajuster automatiquement les limites des axes
    ax1.relim()
    ax1.autoscale_view()

    ax2.relim()
    ax2.autoscale_view()

    ax3.relim()
    ax3.autoscale_view()

# Créer l'objet d'animation
ani = FuncAnimation(fig, update, frames=None, interval=5000)  # L'intervalle est en millisecondes

# Afficher le graphique en temps réel
plt.show()
