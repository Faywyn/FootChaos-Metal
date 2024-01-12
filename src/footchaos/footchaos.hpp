#ifndef footchaos_hpp
#define footchaos_hpp

#include <box2d/box2d.h>
#include <stdio.h>

// -----
// CAR
// -----
class Car {
private:
  b2World *world;
  float steering = 0;

public:
  b2Body *body;

  Car(b2World *world);
  ~Car();

  float getSteering();
  b2Vec2 getOrthogonalSpeed();
  b2Vec2 getNormalSpeed();
  b2Vec2 getPosition();
  b2Vec2 getWorldVector(b2Vec2 vec);

  void setPosition(float x, float y, float angle);

  void tickFriction();
  void tick(float speedControler, float steeringControler);
};

// -----
// BALL
// -----
class Ball {
private:
  b2World *world;

public:
  b2Body *body;

  Ball(b2World *world);
  ~Ball();

  b2Vec2 getOrthogonalSpeed();
  b2Vec2 getNormalSpeed();
  b2Vec2 getPosition();
  b2Vec2 getWorldVector(b2Vec2 vec);

  void setPosition(float x, float y, float angle);

  void tickFriction();
  void tick();
};

// -----
// FOOTCHAOS
// -----
class FootChaos {
private:
  int id;
  b2World *world;

  int sizeOfTeam;
  Car **team1;
  Car **team2;
  Ball *ball;

  b2Body *walls;

public:
  int scoreTeam1 = 0;
  int scoreTeam2 = 0;

  FootChaos(int sizeOfTeam, int id);
  ~FootChaos();

  void tick(float *inputs);
  void resetPosition();

  void setInputs(float *inputs, float *startIndex);
};

#endif /* footchaos_hpp */
