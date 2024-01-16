#include "footchaos.hpp"
#include "../config.hpp"

#include <box2d/box2d.h>
#include <iostream>

// CONST FOR NORMALIZE
float dMaxCenter =
    std::sqrt(FIELD_WIDTH * FIELD_WIDTH + FIELD_LENGHT * FIELD_LENGHT);
float dMaxBall = 2 * dMaxCenter;
float dMaxAdv = 2 * dMaxCenter;
float dMaxGoal = std::sqrt((2 * FIELD_LENGHT) * (2 * FIELD_LENGHT) +
                           FIELD_LENGHT * FIELD_LENGHT);

float normalize(float val, float min, float max, float min_n, float max_n) {
  return ((val - min) / (max - min)) * (max_n - min_n) + min_n;
}

float angle(b2Vec2 A, b2Vec2 B, b2Vec2 dirA) {
  b2Vec2 AB;
  AB.x = B.x - A.x;
  AB.y = B.y - A.y;

  float angle =
      atan2(dirA.x * AB.y - dirA.y * AB.x, dirA.x * AB.x + dirA.y * AB.y);

  return angle;
}

FootChaos::FootChaos(int sizeOfTeam, int id) {
  this->id = id;
  this->sizeOfTeam = sizeOfTeam;

  if (sizeOfTeam % 2 != 0)
    throw std::invalid_argument("Invalid sizeofteam");

  b2Vec2 g(0.0f, 0.0f);
  world = new b2World(g);

  // Creating the walls
  b2BodyDef wallBody;
  walls = world->CreateBody(&wallBody);

  b2Vec2 v1(FIELD_LENGHT, FIELD_WIDTH - LENGTH_ANGLE);
  b2Vec2 v2(FIELD_LENGHT, GOAL_WIDTH);
  b2Vec2 v3(FIELD_LENGHT + BALL_RADIUS * 2.5, +GOAL_WIDTH);
  b2Vec2 v4(FIELD_LENGHT + BALL_RADIUS * 2.5, -GOAL_WIDTH);
  b2Vec2 v5(FIELD_LENGHT, -GOAL_WIDTH);
  b2Vec2 v6(FIELD_LENGHT, -FIELD_WIDTH + LENGTH_ANGLE);
  b2Vec2 v7(FIELD_LENGHT - LENGTH_ANGLE, -FIELD_WIDTH);

  b2Vec2 v8(-FIELD_LENGHT + LENGTH_ANGLE, -FIELD_WIDTH);
  b2Vec2 v9(-FIELD_LENGHT, -FIELD_WIDTH + LENGTH_ANGLE);
  b2Vec2 v10(-FIELD_LENGHT, -GOAL_WIDTH);
  b2Vec2 v11(-FIELD_LENGHT - BALL_RADIUS * 2.5, -GOAL_WIDTH);
  b2Vec2 v12(-FIELD_LENGHT - BALL_RADIUS * 2.5, +GOAL_WIDTH);
  b2Vec2 v13(-FIELD_LENGHT, GOAL_WIDTH);
  b2Vec2 v14(-FIELD_LENGHT, FIELD_WIDTH - LENGTH_ANGLE);
  b2Vec2 v15(-FIELD_LENGHT + LENGTH_ANGLE, FIELD_WIDTH);
  b2Vec2 v16(FIELD_LENGHT - LENGTH_ANGLE, FIELD_WIDTH);

  b2Vec2 vertices[16] = {v1, v2,  v3,  v4,  v5,  v6,  v7,  v8,
                         v9, v10, v11, v12, v13, v14, v15, v16};

  b2ChainShape shape;
  shape.CreateLoop(vertices, 16);
  b2FixtureDef fd;
  fd.shape = &shape;
  fd.density = 0;
  fd.friction = 1;

  walls->CreateFixture(&fd);

  // Creating cars
  team1 = (Car **)malloc(sizeof(Car *) * sizeOfTeam / 2);
  team2 = (Car **)malloc(sizeof(Car *) * sizeOfTeam / 2);
  for (int i = 0; i < sizeOfTeam / 2; i++) {
    team1[i] = new Car(world);
    team2[i] = new Car(world);
  }

  ball = new Ball(world);

  resetPosition();
}

FootChaos::~FootChaos() {
  delete ball;
  for (int i = 0; i < sizeOfTeam / 2; i++) {
    delete team1[i];
    delete team2[i];
  }
  free(team1);
  free(team2);
}

void FootChaos::tick(float *inputs) {
  int indexStart = OUTPUT_LENGTH * sizeOfTeam * id;

  for (int i = 0; i < sizeOfTeam / 2; i++) {
    int indexInput1 = i * OUTPUT_LENGTH + indexStart;
    int indexInput2 =
        i * OUTPUT_LENGTH + indexStart + sizeOfTeam * OUTPUT_LENGTH;

    team1[i]->tick(inputs[indexInput1], inputs[indexInput1 + 1]);
    team2[i]->tick(inputs[indexInput2], inputs[indexInput2 + 1]);
  }
  ball->tick();

  // Check ball pos
  b2Vec2 ballPos = ball->getPosition();
  if (ballPos.x > FIELD_LENGHT + 5) {
    scoreTeam1 += 1;
    resetPosition();
  } else if (ballPos.x < -FIELD_LENGHT - 5) {
    scoreTeam2 += 1;
    resetPosition();
  }
}

void FootChaos::resetPosition() {
  ball->setPosition(0, 0, 0);
  for (int i = 0; i < sizeOfTeam / 2; i++) {
    team1[i]->setPosition(-(FIELD_LENGHT / sizeOfTeam) * (i + 1) / 2, 0,
                          -M_PI / 2);
    team2[i]->setPosition(+(FIELD_LENGHT / sizeOfTeam) * (i + 1) / 2, 0,
                          +M_PI / 2);
  }
}

void FootChaos::setInputs(float *inputs, float *startIndex) {
  // Récupérer la position de la balle
  b2Vec2 ballPos = ball->getPosition();
  b2Vec2 ballVit = ball->body->GetLinearVelocity();

  for (int i = 0; i < sizeOfTeam / 2; i++) {
    int indexInput1 = startIndex[i];
    int indexInput2 = startIndex[i + (sizeOfTeam / 2)];

    b2Vec2 pos1 = team1[i]->getPosition();
    b2Vec2 pos2 = team2[i]->getPosition();
    b2Vec2 ortho1 = team1[i]->getWorldVector(b2Vec2(0, 1));
    b2Vec2 ortho2 = team2[i]->getWorldVector(b2Vec2(0, 1));
    float speed1 = b2Dot(team1[i]->getOrthogonalSpeed(), ortho1);
    float speed2 = b2Dot(team2[i]->getOrthogonalSpeed(), ortho2);

    float dCenter1 = std::sqrt(pos1.x * pos1.x + pos1.y * pos1.y);
    float dCenter2 = std::sqrt(pos2.x * pos2.x + pos2.y * pos2.y);
    float dBall1 = std::sqrt((pos1.x - ballPos.x) * (pos1.x - ballPos.x) +
                             (pos1.y - ballPos.y) * (pos1.y - ballPos.y));
    float dBall2 = std::sqrt((pos2.x - ballPos.x) * (pos2.x - ballPos.x) +
                             (pos2.y - ballPos.y) * (pos2.y - ballPos.y));
    float dGoalA1 = std::sqrt(
        (FIELD_LENGHT - pos1.x) * (FIELD_LENGHT - pos1.x) + pos1.y * pos1.y);
    float dGoalA2 = std::sqrt(
        (FIELD_LENGHT + pos2.x) * (FIELD_LENGHT + pos2.x) + pos2.y * pos2.y);
    float dAdv1 = std::sqrt((pos1.x - pos2.x) * (pos1.x - pos2.x) +
                            (pos1.y - pos2.y) * (pos1.y - pos2.y));
    float dAdv2 = dAdv1;

    float aCenter1 = angle(pos1, b2Vec2(0, 0), ortho1);
    float aCenter2 = angle(pos2, b2Vec2(0, 0), ortho2);
    float aBall1 = angle(pos1, ballPos, ortho1);
    float aBall2 = angle(pos2, ballPos, ortho2);
    float aGoalA1 = angle(pos1, b2Vec2(+1, 0), ortho1);
    float aGoalA2 = angle(pos2, b2Vec2(-1, 0), ortho2);
    float aAdv1 = angle(pos1, pos2, ortho1);
    float aAdv2 = angle(pos2, pos1, ortho2);

    inputs[indexInput1 + 0] = normalize(dCenter1, 0, dMaxCenter, -1, 1);
    inputs[indexInput1 + 1] = normalize(dBall1, 0, dMaxBall, -1, 1);
    inputs[indexInput1 + 2] = normalize(dGoalA1, 0, dMaxGoal, -1, 1);
    inputs[indexInput1 + 3] = normalize(dAdv1, 0, dMaxAdv, -1, 1);
    inputs[indexInput1 + 4] =
        normalize(+pos1.x, -FIELD_LENGHT, FIELD_LENGHT, -1, 1);
    inputs[indexInput1 + 5] =
        normalize(+pos1.y, -FIELD_WIDTH, FIELD_WIDTH, -1, 1);
    inputs[indexInput1 + 6] = normalize(+speed1, -MAX_SPEED, MAX_SPEED, -1, 1);

    inputs[indexInput1 + 7] = std::cos(aCenter1);
    inputs[indexInput1 + 8] = std::sin(aCenter1);
    inputs[indexInput1 + 9] = std::cos(aBall1);
    inputs[indexInput1 + 10] = std::sin(aBall1);
    inputs[indexInput1 + 11] = std::cos(aGoalA1);
    inputs[indexInput1 + 12] = std::sin(aGoalA1);
    inputs[indexInput1 + 13] = std::cos(aAdv1);
    inputs[indexInput1 + 14] = std::sin(aAdv1);

    inputs[indexInput2 + 0] = normalize(dCenter2, 0, dMaxCenter, -1, 1);
    inputs[indexInput2 + 1] = normalize(dBall2, 0, dMaxBall, -1, 1);
    inputs[indexInput2 + 2] = normalize(dGoalA2, 0, dMaxGoal, -1, 1);
    inputs[indexInput2 + 3] = normalize(dAdv2, 0, dMaxAdv, -1, 1);
    inputs[indexInput2 + 4] =
        normalize(-pos2.x, -FIELD_LENGHT, FIELD_LENGHT, -1, 1);
    inputs[indexInput2 + 5] =
        normalize(-pos2.y, -FIELD_WIDTH, FIELD_WIDTH, -1, 1);
    inputs[indexInput2 + 6] = normalize(-speed2, -MAX_SPEED, MAX_SPEED, -1, 1);

    inputs[indexInput2 + 7] = std::cos(aCenter2);
    inputs[indexInput2 + 8] = std::sin(aCenter2);
    inputs[indexInput2 + 9] = std::cos(aBall2);
    inputs[indexInput2 + 10] = std::sin(aBall2);
    inputs[indexInput2 + 11] = std::cos(aGoalA2);
    inputs[indexInput2 + 12] = std::sin(aGoalA2);
    inputs[indexInput2 + 13] = std::cos(aAdv2);
    inputs[indexInput2 + 14] = std::sin(aAdv2);
  }
}
