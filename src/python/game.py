import csv
from math import floor, cos, sin, pi
import pygame
from pygame.locals import *
import sys
import os

# path = ("~/DevWorkspace/GitHub/FootChaos-Metal/trainings/")
path = (os.getcwd() + "/trainings/")

trainingId = int(sys.argv[1])
matchNb = int(sys.argv[2] or -1)
maxNb = int(sys.argv[3] or -1)

# Afficher en  continue
continusPrint = (maxNb != (-1))

RATIO: float = 10 * 1/9
BORDER: int = 10

TICKS_COEF = 1.5

red = (224, 59, 4)
blue = (45, 83, 207)

def printCar(x, y, angle, steering, color):
    # Les voitures sont des rectangles de longueur CAR_LENGTH et de largeur CAR_WIDTH
    # On va donc calculer les coordonnées des 3 sommets du triangle

    x = float(x) + CAR_WIDTH / 2 + BORDER/2
    y = float(y) + CAR_LENGTH / 2 - BORDER/2
    angle = float(angle) + pi / 2

    # Coordonnées du centre haut et bas de la voiture
    xH = x + RATIO * CAR_LENGTH/2 * cos(angle)
    yH = y + RATIO * CAR_LENGTH/2 * sin(angle)
    xB = x - RATIO * CAR_LENGTH/2 * cos(angle)
    yB = y - RATIO * CAR_LENGTH/2 * sin(angle)

    # Coordonnées des 4 sommets de la voiture
    x1 = xH + RATIO * CAR_WIDTH/2 * sin(angle)    # Haut gauche
    y1 = yH - RATIO * CAR_WIDTH/2 * cos(angle)
    x2 = xH - RATIO * CAR_WIDTH/2 * sin(angle)    # Haut droite
    y2 = yH + RATIO * CAR_WIDTH/2 * cos(angle)
    x3 = xB + RATIO * CAR_WIDTH/2 * sin(angle)    # Bas gauche
    y3 = yB - RATIO * CAR_WIDTH/2 * cos(angle)
    x4 = xB - RATIO * CAR_WIDTH/2 * sin(angle)    # Bas droit
    y4 = yB + RATIO * CAR_WIDTH/2 * cos(angle)

    # Afficher les roues
    def drawWheel(x_, y_, a):
        longueur = RATIO * CAR_LENGTH/10
        largeur = RATIO * CAR_WIDTH / 6
        xH = x_ + longueur * cos(a)
        yH = y_ + longueur * sin(a)
        xB = x_ - longueur * cos(a)
        yB = y_ - longueur * sin(a)

        x1 = xH + largeur * sin(a)
        y1 = yH - largeur * cos(a)
        x2 = xH - largeur * sin(a)
        y2 = yH + largeur * cos(a)
        x3 = xB + largeur * sin(a)
        y3 = yB - largeur * cos(a)
        x4 = xB - largeur * sin(a)
        y4 = yB + largeur * cos(a)

        pygame.draw.polygon(screen, "black", [(x2, y2), (x1, y1), (x3, y3), (x4, y4)])

    r1 = [ x1 * 0.85 + x3 * 0.15, y1 * 0.85 + y3 * 0.15 ]
    r2 = [ x2 * 0.85 + x4 * 0.15, y2 * 0.85 + y4 * 0.15 ]
    r3 = [ x1 * 0.15 + x3 * 0.85, y1 * 0.15 + y3 * 0.85 ]
    r4 = [ x2 * 0.15 + x4 * 0.85, y2 * 0.15 + y4 * 0.85 ]


    drawWheel(r1[0], r1[1], angle + steering * 0.5)
    drawWheel(r2[0], r2[1], angle + steering * 0.5)
    drawWheel(r3[0], r3[1], angle)
    drawWheel(r4[0], r4[1], angle)


    # Afficher la voiture
    pygame.draw.polygon(screen, color, [(x1, y1), (x2, y2), (x4, y4), (x3, y3)])


    # Afficher les fenêtres de la voiture
    # Coordonnées du centre haut et bas de la voiture
    xfH = xH * 0.9 + xB * 0.1
    yfH = yH * 0.9 + yB * 0.1
    xfB = xB * 0.45 + xH * 0.55
    yfB = yB * 0.45 + yH * 0.55
    xf1 = xfH + RATIO * CAR_WIDTH * 2/5 * sin(angle)
    yf1 = yfH - RATIO * CAR_WIDTH * 2/5 * cos(angle)
    xf2 = xfH - RATIO * CAR_WIDTH * 2/5 * sin(angle)
    yf2 = yfH + RATIO * CAR_WIDTH * 2/5 * cos(angle)
    xf3 = xfB + RATIO * CAR_WIDTH * 2/5 * sin(angle)
    yf3 = yfB - RATIO * CAR_WIDTH * 2/5 * cos(angle)
    xf4 = xfB - RATIO * CAR_WIDTH * 2/5 * sin(angle)
    yf4 = yfB + RATIO * CAR_WIDTH * 2/5 * cos(angle)

    # Afficher les fenêtres
    pygame.draw.polygon(screen, "white", [(xf1, yf1), (xf2, yf2), (xf4, yf4), (xf3, yf3)])

class Coord():
    x: float    # Coordonnées réelles
    y: float
    xS: int     # Coordonnées fenêtre
    yS: int
    angle: float = 0
    steering: float = 0
    symetry: bool = False


    def __init__(self, x: str, y: str, angle: str = None, steering: str = None, symetry: bool = False):
        self.x = float(x)
        self.y = float(y)

        if angle is not None:
            self.angle = float(angle)
        if steering is not None:
            self.steering = float(steering)
        
        if symetry:
            self.symetrie()
            return

        self.xS = floor((self.x + LENGTH) * RATIO)
        self.yS = floor((self.y + WIDTH) * RATIO)

    def symetrie(self):
        self.xS = floor((LENGTH - self.x) * RATIO)
        self.yS = floor((WIDTH - self.y) * RATIO)
        self.angle = self.angle + pi
        self.steering = -self.steering

while ((matchNb < maxNb) or not continusPrint):
    if (continusPrint):
        pathFile = path + str(trainingId) + "/M" + str(matchNb) + ".csv"
        matchNb += 10
        print (pathFile)
    else:
        pathFile = path

    print(pathFile)
    with open(pathFile, 'r') as file:
        csvreader = list(csv.reader(file, delimiter=";"))

        infoPartie = interestingrows=[row for idx, row in enumerate(csvreader) if idx in (0, 0)][0]
        # LENGTH_TERRAIN; WIDTH_TERRAIN; GOAL_WIDTH; TICKS_SECOND; BALL_RADIUS; GAME_LENGTH; NMB_JOUEURS_EQUIPE_1;
        # NMB_JOIUEURS_EQUIPE_2; CAR_LENGTH; CAR_WIDTH
        LENGTH = float(infoPartie[0])
        WIDTH = float(infoPartie[1])
        GOAL_WIDTH = float(infoPartie[2])
        BALL_RADIUS = float(infoPartie[3])
        TICKS_SECOND = int(infoPartie[4])
        GAME_LENGTH = int(infoPartie[5])
        TEAM_SIZE_1 = int(infoPartie[6])
        TEAM_SIZE_2 = int(infoPartie[7])
        CAR_LENGTH = float(infoPartie[8])
        CAR_WIDTH = float(infoPartie[9])
        ANGLE_WIDTH = 115#float(infoPartie[10])

        pygame.init()
        pygame.font.init()
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        flags = DOUBLEBUF
        screen = pygame.display.set_mode((RATIO * (LENGTH * 2) + 2 * BORDER, RATIO * (WIDTH * 2) + 2 * BORDER), flags)
        clock = pygame.time.Clock()
        exe = True

        if (matchNb != -1):
            pygame.display.set_caption("Training " + str(trainingId) + " - Match " + str(matchNb - 10))
        else:
            pygame.display.set_caption("FootChaos")

        while exe:
            # TEAM1SCORE; TEAM2SCORE; X_BALL; Y_BALL; X_1; Y_1; A_1; V_1; X_2; Y_2; A_2; V_2; ..., X_N; Y_N; A_N; V_N
            tick = -1
            exe = not continusPrint
            for ligne in csvreader:
                tick += 1

                steerings = [0 in range(TEAM_SIZE_1 + TEAM_SIZE_2)]

                if (tick == 0):
                    continue

                #if (tick % TICKS_COEF != 0):
                #    continue

                if (tick % TICKS_SECOND == 0):
                    print(ligne)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exe = False
                        pygame.quit()
                        break

                if (ligne == ['0' for i in range((len(ligne)))]):
                    exe = False
                    pygame.quit()
                    break


                # Effacez l'écran
                screen.fill("white")

                # Afficher le terrain (10 bande vert)
                for i in range(0, 5):
                    c1 = (11, 138, 49)
                    c2 = (21, 148, 59)
                    screen.fill(c1, pygame.Rect(
                        BORDER + floor((2 * i * LENGTH / 5) * RATIO),
                        BORDER,
                        floor(RATIO * (LENGTH / 5)) + 1,
                        floor(RATIO * WIDTH * 2)
                    ))
                    screen.fill(c2, pygame.Rect(
                        BORDER + floor(((2 * i + 1) * LENGTH / 5) * RATIO),
                        BORDER,
                        floor(RATIO * (LENGTH / 5)) + 1,
                        floor(RATIO * WIDTH * 2)
                    ))

                # Afficher les bordures
                couleur = "black"
                screen.fill(couleur, pygame.Rect(0, 0, BORDER, RATIO * (WIDTH * 2) + 2 * BORDER))
                screen.fill(couleur, pygame.Rect(0, 0, RATIO * (LENGTH * 2) + 2 * BORDER, BORDER))
                screen.fill(couleur, pygame.Rect(RATIO * (LENGTH * 2) + BORDER, 0, BORDER, RATIO * (WIDTH * 2) + 2 * BORDER))
                screen.fill(couleur, pygame.Rect(0, RATIO * (WIDTH * 2) + BORDER, RATIO * (LENGTH * 2) + 2 * BORDER, BORDER))

                pygame.draw.polygon(screen, couleur, [
                    (BORDER, BORDER + ANGLE_WIDTH * RATIO),
                    (BORDER + ANGLE_WIDTH * RATIO, BORDER),
                    (BORDER, BORDER)
                ])
                pygame.draw.polygon(screen, couleur, [
                    (BORDER + 2 * LENGTH * RATIO, BORDER + ANGLE_WIDTH * RATIO),
                    (BORDER + (2 * LENGTH - ANGLE_WIDTH) * RATIO, BORDER),
                    (2 * LENGTH * RATIO + BORDER, 0)
                ])
                pygame.draw.polygon(screen, couleur, [
                    (BORDER, BORDER + (WIDTH * 2 - ANGLE_WIDTH) * RATIO),
                    (BORDER + ANGLE_WIDTH * RATIO, RATIO * WIDTH * 2 + BORDER),
                    (BORDER, 2 * WIDTH * RATIO + BORDER)
                ])
                pygame.draw.polygon(screen, couleur, [
                    (BORDER + 2 * LENGTH * RATIO, BORDER + (2 * WIDTH - ANGLE_WIDTH) * RATIO),
                    (BORDER + RATIO * (2 * LENGTH - ANGLE_WIDTH), BORDER + 2 * WIDTH * RATIO),
                    (BORDER + LENGTH * 2 * RATIO, BORDER + WIDTH * 2 * RATIO)
                ])

                # Afficher le score
                text_surface = my_font.render(str(ligne[0]) + " - " + str(ligne[1]), False, (255, 255, 255))
                screen.blit(text_surface, (2 * BORDER, BORDER))

                # Afficher la barre de progression
                screen.fill("grey", pygame.Rect(BORDER, BORDER / 2,  (2 * LENGTH * RATIO) * (tick / (GAME_LENGTH * TICKS_SECOND)), 5))

                # Afficher les buts
                screen.fill(red, pygame.Rect(0, RATIO * (WIDTH - GOAL_WIDTH) + BORDER, BORDER, RATIO * (GOAL_WIDTH * 2)))
                screen.fill(blue, pygame.Rect(RATIO * (LENGTH * 2) + BORDER, RATIO * (WIDTH - GOAL_WIDTH) + BORDER, BORDER, RATIO * (GOAL_WIDTH * 2)))

                # Afficher la balle
                coordBall = Coord(ligne[2], ligne[3])
                pygame.draw.circle(screen, "black", (coordBall.xS + BORDER, coordBall.yS + BORDER), BALL_RADIUS * RATIO)
                pygame.draw.circle(screen, "white", (coordBall.xS + BORDER, coordBall.yS + BORDER), BALL_RADIUS * RATIO * (1 - 1/4))

                # Afficher les voitures de l'équipe 1
                for i in range(0, TEAM_SIZE_1):
                    pos = Coord(ligne[4 * i + 4], ligne[4 * i + 5], ligne[4 * i + 6], ligne[4 * i + 7], False)
                    printCar(pos.xS, pos.yS, pos.angle, pos.steering, red)

                # Afficher les voitures de l'équipe 2
                for i in range(0, TEAM_SIZE_2):
                    pos = Coord(ligne[4 * i + 4 + TEAM_SIZE_2 * 4], ligne[4 * i + 5 + TEAM_SIZE_2 * 4],
                                ligne[4 * i + 6 + TEAM_SIZE_2 * 4], ligne[4 * i + 7 + TEAM_SIZE_2 * 4],
                                False)
                    printCar(pos.xS, pos.yS, pos.angle, pos.steering, blue)

                # pygame.draw.line(screen, "red", (0, WIDTH * RATIO + BORDER), (2 * LENGTH * RATIO + BORDER, WIDTH * RATIO + BORDER))

                # Affichez le contenu de l'écran
                pygame.display.flip()

                clock.tick(TICKS_SECOND * TICKS_COEF)

        pygame.quit()
