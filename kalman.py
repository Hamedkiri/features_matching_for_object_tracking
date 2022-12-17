class Compteur:
    i = 0
    j = 0
    key = []

    def som(cls):
        cls.i += 1

    som = classmethod(som)


# ------------------------


import cv2
from tkinter import *
import time
import numpy as np
from turtle import Vec2D

WIDTH = 2000
HEIGHT = 3000
# COTE=20
COTE = 10

root = Tk()

cnv = Canvas(root, width=WIDTH, height=HEIGHT, background="ivory")
cnv.grid(row=1, column=0)

DIR = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}
kalman = cv2.KalmanFilter(3, 3, 0, cv2.CV_64F)

# Matrice de transition, par défaut identité car il faut nécessairement une commande pour mettre le véhicule en mouvement: kalman.A. A est une matrice carré d'ordre le nombre d'états.
kalman.transitionMatrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
# Matrice de controle lié aux commandes pour mettre l'objet en mouvement: kalman.B. B est une matrice dont le nombre de ligne est le nombre d'états et le nombre de colonnes le nombre d'entrées de contrôles.

# Matrix de covariance du bruit d'etat: kalman.Q. C'est une carré d'ordre le nombre d'états.
kalman.processNoiseCov = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

kalman.errorCovPre = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

# Matrice de mesure, noté H, on aurait pu écrire: kalman.H. Le nombre de lignes de H est le nombre d'observations du capteur et son nombre de colonnes le nombres d'états
kalman.measurementMatrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
# Matrice de bruit du capteur: kalman.R. Est une matrice carré d'ordre le nombre de mesures du capteur.
kalman.measurementNoiseCov = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

# Etat après correcion
kalman.statePost = 0.01 * np.array([[0], [0], [0]])
# Matrice de covariance après correction
kalman.errorCovPost = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


def bouge(event):
    Compteur.i = time.time()
    key = event.keysym
    Compteur.j = time.time()

    Compteur.key.append(key)
    dx, dy = DIR[key]
    dx_cm = (10 * dx * 8.47) / 1000
    dy_cm = (10 * dy * 16.93) / 2000
    a, b, segment = pile[-1]
    a_cm = (a * 8.47) / 1000
    b_cm = (b * 16.93) / 2000
    dt = Compteur.j - Compteur.i
    v = np.sqrt((0.01 * dx_cm) ** 2 + (0.01 * dy_cm) ** 2) / dt

    def angle():

        if b == 0:
            return 0

        elif a == 0:
            return 90
        elif a > 0:
            angle = (np.arctan(b / a) * 180) / np.pi
            return angle
        elif a < 0 and b > 0:
            angle = ((np.pi - np.arctan(b / a)) * 180) / np.pi
            return
        elif a < 0 and b < 0:
            angle = ((np.pi + np.arctan(b / a)) * 180) / np.pi
            return angle

    angles = angle()

    position = cnv.coords(tr)
    P = np.array([position[4], position[5]])
    Q = np.array([position[2], position[3]])
    R = np.array([position[0], position[1]])

    def rot(C, t, M):
        CM = Vec2D(*M) - Vec2D(*C)
        return Vec2D(*C) + CM.rotate(t)

    def rotate(t):
        C = (P + Q + R) / 3

        PP = rot(C, float(t), P)
        QQ = rot(C, float(t), Q)
        RR = rot(C, float(t), R)
        cnv.coords(tr, *PP, *QQ, *RR)

    n = len(Compteur.key)
    angla = 0
    if Compteur.key[n - 2] != Compteur.key[n - 1]:
        if Compteur.key[n - 2] == "Right":
            if Compteur.key[n - 1] == "Down":
                rotate(90)
                angla = np.pi / 2
            elif Compteur.key[n - 1] == 'Left':
                rotate(180)
                angla = np.pi
            elif Compteur.key[n - 1] == 'Up':
                rotate(-90)
                angla = -np.pi
        if Compteur.key[n - 2] == "Left":
            if Compteur.key[n - 1] == "Down":
                rotate(-90)
                angla = -np.pi
            elif Compteur.key[n - 1] == 'Right':
                rotate(180)
                angla = np.pi
            elif Compteur.key[n - 1] == 'Up':
                rotate(90)
                angla = np.pi / 2

        if Compteur.key[n - 2] == "Down":
            if Compteur.key[n - 1] == "Up":
                rotate(180)
                angla = np.pi
            elif Compteur.key[n - 1] == 'Right':
                rotate(-90)
                angla = -np.pi / 2
            elif Compteur.key[n - 1] == 'Left':
                rotate(90)
                angla = np.pi / 2

        if Compteur.key[n - 2] == "Up":
            if Compteur.key[n - 1] == "Down":
                rotate(180)
            elif Compteur.key[n - 1] == 'Right':
                rotate(90)
            elif Compteur.key[n - 1] == 'Left':
                rotate(-90)

    # -----------------------------

    wo = angla / dt
    u = np.array([[v], [wo]])
    v_err = np.random.random(3).reshape(-1, 1)
    kalman.controlMatrix = np.array([[np.cos(angla) * dt, 0], [np.sin(angla) * dt, 0], [0, dt]]) + v_err
    kalman.statePre = kalman.transitionMatrix @ kalman.statePost + kalman.controlMatrix @ u
    kalman.errorCovPre = kalman.transitionMatrix @ kalman.errorCovPre @ kalman.transitionMatrix + kalman.processNoiseCov
    w = np.array([[0.007], [0.007], [0.007]])
    measurement_pre = kalman.measurementMatrix @ kalman.statePre + w
    measurement = np.array([[a], [b], [angles]])
    # measurement_res=measurement-measurement_pre

    # Innov=kalman.measurementMatrix@kalman.errorCovPre@kalman.measurementMatrix+kalman.measurementNoiseCov

    # kalman.gain=kalman.errorCovPre@kalman.measurementMatrix@np.linalg.inv(Innov)

    # ---------------------------

    segment = cnv.create_line(a * COTE + COTE // 2, b * COTE + COTE // 2, a * COTE + COTE // 2 + dx * COTE,
                              b * COTE + COTE // 2 + dy * COTE, fill='blue', width=4, capstyle=ROUND)
    pile.append([a + dx, b + dy, segment])

    text = cnv.create_text((900, 800), text="(" + str(a + dx) + "," + str(b + dy) + ")" + "--" + "--" + str(
        round(np.sqrt((dx_cm + a_cm) ** 2 + (dy_cm + b_cm) ** 2), 2)) + "cm" + "--" + str(
        np.ceil(v)) + "m/s" + "--" + str(angles) + "°", fill="red", font="Arial 30 bold")

    def effacer(ident):
        cnv.delete(ident)

    cnv.after(150, effacer, text)

    # kalman.statePost=kalman.statePre+kalman.gain@measurement_res
    prediction = kalman.correct(0.99999 * measurement)
    segment2 = cnv.create_line(int(prediction[0]) * COTE + COTE // 2, int(prediction[1]) * COTE + COTE // 2,
                               int(prediction[0]) * COTE + COTE // 2 + dx * COTE,
                               int(prediction[1]) * COTE + COTE // 2 + dy * COTE, fill='red', width=4, capstyle=ROUND)

    # ---------------------------

    cnv.move(tr, dx * 10, dy * 10)

    cnv.after(150, effacer, segment)


tr = cnv.create_polygon((0, 0), (0, 20), (10, 10), width=8, fill="blue", tag='triangle')

pile = [(0, 0, tr)]

for key in ["<Left>", "<Right>", "<Up>", "<Down>"]:
    root.bind(key, bouge)

root.mainloop()