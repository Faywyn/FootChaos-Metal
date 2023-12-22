#ifndef config_h
#define config_h

// ----------
// GAME CONFIG
// ----------

// Field config
#define FIELD_LENGHT 512.0f
#define FIELD_WIDTH 409.6f
#define GOAL_WIDTH 89.3f
#define LENGTH_ANGLE 115.2f // Diagonal walls

// Item config
#define BALL_RADIUS 10.0f
#define CAR_LENGHT 30.0f
#define CAR_WIDTH 15.0f

// Forces config
#define FRICTION_COEF 5.0f

#define MAX_LATERAL_IMPULSE (CAR_LENGHT * CAR_WIDTH * 50)
#define MAX_FORCE (CAR_LENGHT * CAR_WIDTH * 10)
#define MAX_SPEED 25.0f
#define MAX_STEERING_SPEED 0.1f
#define TORQUE_FORCE (CAR_LENGHT * CAR_WIDTH * 1e2)

// Game length
#define TICK_DURATION_COEF 2
#define TICKS_SECOND 60
#define GAME_LENGTH (30 * TICK_DURATION_COEF) // s

// ----------
// NETWORKS CONFIG
// ----------

#define MIN_WEIGHT (-10)
#define MAX_WEIGHT (+10)

#define INPUT_LENGTH 15
#define OUTPUT_LENGTH 2

#endif /* config_h */
