#pragma once

// ----- GAME CONFIG -----
#define FIELD_LENGHT 512.0f
#define FIELD_WIDTH 409.6f
#define GOAL_WIDTH 89.3f
#define LENGTH_ANGLE 115.2f // Diagonal walls

#define BALL_RADIUS 10.0f
#define CAR_LENGHT 30.0f
#define CAR_WIDTH 15.0f

#define FRICTION_COEF 5.0f

#define MAX_LATERAL_IMPULSE (CAR_LENGHT * CAR_WIDTH * 50)
#define MAX_FORCE (CAR_LENGHT * CAR_WIDTH * 100)
#define MAX_SPEED 25.0f
#define MAX_STEERING_SPEED 0.1f
#define TORQUE_FORCE (CAR_LENGHT * CAR_WIDTH * 1e2)

#define TICKS_SECOND 60
#define GAME_LENGTH 60 // s

#define DATA_PER_PLAYER 4
// ----- GAME CONFIG -----

// ----- NETWORKS CONFIG -----
#define INPUT_LENGTH 15
#define OUTPUT_LENGTH 2
#define INPUT_NORM_DATA_LENGTH 7
#define INPUT_TRIG_DATA_LENGTH 4
#define DEFAULT_WEIGHT_VALUE 0
#define WEIGHT_RADIUS 2 // Gaussian generation

#define NB_LAYER 13
const int NB_NEURON_PER_LAYER[NB_LAYER] = {
    INPUT_LENGTH, 15, 15, 15, 15, 15, 14, 12, 10, 8, 6, 4, OUTPUT_LENGTH};
// ----- NETWORKS CONFIG -----

// ----- TRAINING CONFIG -----
#define TRAININGS_PATH (fs::current_path() / "trainings")

#define NEW_BLOOD_COEF 0.05f
#define COPY_COEF 0.01f
#define NB_WEIGHT_CHANGE 25
// ----- TRAINING CONFIG -----

// ----- COLORS -----
#define RESET "\033[0m"
#define RED "\033[31m"
#define CYAN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define PINK "\033[36m"
#define GREEN "\033[37m"
// ----- COLORS -----

// ----- OTHER -----
#define NB_THREAD 32
#define STAT_TAB_START 4
#define NB_STATS 10
// ----- OTHER -----
