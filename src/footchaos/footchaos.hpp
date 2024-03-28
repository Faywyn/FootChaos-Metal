#pragma once

#include <box2d/box2d.h>
#include <filesystem>
#include <stdio.h>

namespace fs = std::filesystem;

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
  int tickId = 0;
  bool idle = false;
  int idleTick = 0;
  int farTick = 0;
  b2World *world;

  float **data;
  bool save;
  fs::path path;

  Car *team1;
  Car *team2;
  Ball *ball;

  b2Body *walls;
  bool random;

public:
  int scoreTeam1 = 0;
  int scoreTeam2 = 0;
  float scoreTeam1Pos = 0;
  float scoreTeam2Pos = 0;

  FootChaos(int id, bool random, fs::path chemin);
  ~FootChaos();

  void tick(float *inputs);
  void resetPosition();
  void resetGame();

  void setInputs(float *inputsDataNorm, float *inputsDataTrig, int startIndex);
  void addData(float stearing1, float stearing2);

  void checkIdle();
};
