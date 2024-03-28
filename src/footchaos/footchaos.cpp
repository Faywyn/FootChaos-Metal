#include "footchaos.hpp"
#include "../config.hpp"
#include "utils.hpp"

#include <box2d/box2d.h>
#include <fstream>
#include <iostream>
#include <unistd.h>

// CONST FOR NORMALIZE FUNCTION
float dMaxCenter =
    std::sqrt(FIELD_WIDTH * FIELD_WIDTH + FIELD_LENGHT * FIELD_LENGHT);
float dMaxBall = 2 * dMaxCenter;
float dMaxAdv = 2 * dMaxCenter;
float dMaxGoal = std::sqrt((2 * FIELD_LENGHT) * (2 * FIELD_LENGHT) +
                           FIELD_LENGHT * FIELD_LENGHT);

/// Normalize a value, in order to be between min_n and max_n instead of min and
/// max
/// Parameters:
///  - val
///  - min
///  - max
///  - min_n
///  - max_n
float normalize(float val, float min, float max, float min_n, float max_n) {
  return ((val - min) / (max - min)) * (max_n - min_n) + min_n;
}

/// Get the angle between 2 vector depending on the direction
/// Parameters:
///  - A (first vector)
///  - B (second vector)
///  - dirA
float angle(b2Vec2 A, b2Vec2 B, b2Vec2 dirA) {
  b2Vec2 AB;
  AB.x = B.x - A.x;
  AB.y = B.y - A.y;

  float angle =
      atan2(dirA.x * AB.y - dirA.y * AB.x, dirA.x * AB.x + dirA.y * AB.y);

  return angle;
}

/// Class constructor
/// Parameters:
///  - sizeOfTeam (num of car)
///  - id
///  - path ("" for no save)
FootChaos::FootChaos(int id, bool random, fs::path path) {
  this->id = id;
  this->path = path;
  this->save = path != "";
  this->random = random;

  // Malloc data if needed for save
  if (save) {
    int nbTicks = TICKS_SECOND * GAME_LENGTH;

    this->data = (float **)malloc(sizeof(float *) * nbTicks);

    for (int i = 0; i < nbTicks; i++) {
      this->data[i] =
          (float *)malloc(sizeof(float) * (DATA_PER_PLAYER * 2 + 4));
    }
  }

  // Create the world
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
  team1 = new Car(world);
  team2 = new Car(world);

  // Create the ball
  ball = new Ball(world);

  resetPosition();
}

/// FootChaos destructor
FootChaos::~FootChaos() {
  delete ball;
  delete team1;
  delete team2;
  delete world;

  // Save if needed
  if (save) {
    std::ofstream csv(path);
    csv << FIELD_LENGHT << ";" << FIELD_WIDTH << ";" << GOAL_WIDTH << ";"
        << BALL_RADIUS << " ; " << TICKS_SECOND << ";" << GAME_LENGTH << ";"
        << 1 << ";" << 1 << ";" << CAR_LENGHT << ";" << CAR_WIDTH << ";"
        << LENGTH_ANGLE << "\n";

    int nbTicks = GAME_LENGTH * TICKS_SECOND;
    int nbData = 2 * DATA_PER_PLAYER + 4;
    for (int j = 0; j < nbTicks; j++) {
      for (int k = 0; k < nbData; k++) {
        csv << data[j][k] << (k == nbData - 1 ? "\n" : ";");
      }
      free(data[j]);
    }
    csv.close();
    free(data);
  }
}

/// Perform a tick in the game
/// Parameters:
///  - *inputs (inputs of the cars)
void FootChaos::tick(float *inputs) {
  if (idle)
    return;
  checkIdle();

  // Tick every car
  int indexStart = 2 * OUTPUT_LENGTH * id;
  int indexInput1 = indexStart;
  int indexInput2 = indexStart + OUTPUT_LENGTH;

  team1->tick(inputs[indexInput1], inputs[indexInput1 + 1]);
  team2->tick(inputs[indexInput2], inputs[indexInput2 + 1]);

  // Tick ball
  ball->tick();

  // Check ball pos (and so goals)
  b2Vec2 ballPos = ball->getPosition();
  if (ballPos.x > FIELD_LENGHT + 5) {
    scoreTeam1 += 1;
    resetPosition();
  } else if (ballPos.x < -FIELD_LENGHT - 5) {
    scoreTeam2 += 1;
    resetPosition();
  }

  // PosScore
  float score =
      pow(ballPos.x / FIELD_LENGHT, 3) * (1 - pow(ballPos.y / FIELD_WIDTH, 2));
  scoreTeam1Pos += score;
  scoreTeam2Pos -= score;

  // Perform global world tick
  int32 velocityIterations = 6;
  int32 positionIterations = 2;
  float timeTick = 5.0f / TICKS_SECOND;
  world->Step(timeTick, velocityIterations, positionIterations);

  if (save)
    addData(team1->getSteering(), team2->getSteering());

  tickId++;
}

/// Reset position
void FootChaos::resetPosition() {
  ball->setPosition(randomFloat(), randomFloat(), randomFloat());
  if (random == false) {
    team1->setPosition(-FIELD_LENGHT / 2, 0, -M_PI / 2);
    team2->setPosition(+FIELD_LENGHT / 2, 0, +M_PI / 2);
    return;
  }
  float r1 = 1.0f / 2; // abs(randomFloat()) * 0.7f;
  float r2 = randomFloat() * 0.9f;
  float r3 = -1.0f / 2;

  team1->setPosition(-r1 * FIELD_LENGHT, +r2 * FIELD_WIDTH, r3 * M_PI);
  team2->setPosition(+r1 * FIELD_LENGHT, -r2 * FIELD_WIDTH, r3 * M_PI - M_PI);
}

void FootChaos::resetGame() {
  resetPosition();
  scoreTeam1 = 0;
  scoreTeam2 = 0;
  scoreTeam1Pos = 0;
  scoreTeam2Pos = 0;
  idle = false;
  tickId = 0;
  idleTick = 0;
  farTick = 0;
}

/// Set the inputs for the networks
/// Parameters:
///  - *inputs (tab to edit)
///  - *startIndex (where to edit inputs)
void FootChaos::setInputs(float *inputsDataNorm, float *inputsDataTrig,
                          int startIndex) {
  // Get ball data
  b2Vec2 ballPos = ball->getPosition();

  b2Vec2 pos1 = team1->getPosition();
  b2Vec2 pos2 = team2->getPosition();
  b2Vec2 ortho1 = team1->getWorldVector(b2Vec2(0, 1));
  b2Vec2 ortho2 = team2->getWorldVector(b2Vec2(0, 1));
  float speed1 = b2Dot(team1->getOrthogonalSpeed(), ortho1);
  float speed2 = b2Dot(team2->getOrthogonalSpeed(), ortho2);

  float dCenter1 = std::sqrt(pos1.x * pos1.x + pos1.y * pos1.y);
  float dCenter2 = std::sqrt(pos2.x * pos2.x + pos2.y * pos2.y);
  float dBall1 = std::sqrt((pos1.x - ballPos.x) * (pos1.x - ballPos.x) +
                           (pos1.y - ballPos.y) * (pos1.y - ballPos.y));
  float dBall2 = std::sqrt((pos2.x - ballPos.x) * (pos2.x - ballPos.x) +
                           (pos2.y - ballPos.y) * (pos2.y - ballPos.y));
  float dGoalA1 = std::sqrt((FIELD_LENGHT - pos1.x) * (FIELD_LENGHT - pos1.x) +
                            pos1.y * pos1.y);
  float dGoalA2 = std::sqrt((FIELD_LENGHT + pos2.x) * (FIELD_LENGHT + pos2.x) +
                            pos2.y * pos2.y);
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

  int index1N = (startIndex * 2 + 0) * INPUT_NORM_DATA_LENGTH * 3;
  int index1T = (startIndex * 2 + 0) * INPUT_TRIG_DATA_LENGTH;
  inputsDataNorm[index1N + 0] = dCenter1;
  inputsDataNorm[index1N + 1] = 0;
  inputsDataNorm[index1N + 2] = dMaxCenter;

  inputsDataNorm[index1N + 3] = dBall1;
  inputsDataNorm[index1N + 4] = 0;
  inputsDataNorm[index1N + 5] = dMaxBall;

  inputsDataNorm[index1N + 6] = dGoalA1;
  inputsDataNorm[index1N + 7] = 0;
  inputsDataNorm[index1N + 8] = dMaxGoal;

  inputsDataNorm[index1N + 9] = dAdv1;
  inputsDataNorm[index1N + 10] = 0;
  inputsDataNorm[index1N + 11] = dMaxAdv;

  inputsDataNorm[index1N + 12] = pos1.x;
  inputsDataNorm[index1N + 13] = -FIELD_LENGHT;
  inputsDataNorm[index1N + 14] = FIELD_LENGHT;

  inputsDataNorm[index1N + 15] = pos1.y;
  inputsDataNorm[index1N + 16] = -FIELD_WIDTH;
  inputsDataNorm[index1N + 17] = FIELD_WIDTH;

  inputsDataNorm[index1N + 18] = speed1;
  inputsDataNorm[index1N + 19] = -MAX_SPEED;
  inputsDataNorm[index1N + 20] = MAX_SPEED;

  inputsDataTrig[index1T + 0] = aCenter1;
  inputsDataTrig[index1T + 1] = aBall1;
  inputsDataTrig[index1T + 2] = aGoalA1;
  inputsDataTrig[index1T + 3] = aAdv1;

  int index2N = (startIndex * 2 + 1) * INPUT_NORM_DATA_LENGTH * 3;
  int index2T = (startIndex * 2 + 1) * INPUT_TRIG_DATA_LENGTH;
  inputsDataNorm[index2N + 0] = dCenter2;
  inputsDataNorm[index2N + 1] = 0;
  inputsDataNorm[index2N + 2] = dMaxCenter;

  inputsDataNorm[index2N + 3] = dBall2;
  inputsDataNorm[index2N + 4] = 0;
  inputsDataNorm[index2N + 5] = dMaxBall;

  inputsDataNorm[index2N + 6] = dGoalA2;
  inputsDataNorm[index2N + 7] = 0;
  inputsDataNorm[index2N + 8] = dMaxGoal;

  inputsDataNorm[index2N + 9] = dAdv2;
  inputsDataNorm[index2N + 10] = 0;
  inputsDataNorm[index2N + 11] = dMaxAdv;

  inputsDataNorm[index2N + 12] = -pos2.x;
  inputsDataNorm[index2N + 13] = -FIELD_LENGHT;
  inputsDataNorm[index2N + 14] = FIELD_LENGHT;

  inputsDataNorm[index2N + 15] = -pos2.y;
  inputsDataNorm[index2N + 16] = -FIELD_WIDTH;
  inputsDataNorm[index2N + 17] = FIELD_WIDTH;

  inputsDataNorm[index2N + 18] = speed2;
  inputsDataNorm[index2N + 19] = -MAX_SPEED;
  inputsDataNorm[index2N + 20] = MAX_SPEED;

  inputsDataTrig[index2T + 0] = aCenter2;
  inputsDataTrig[index2T + 1] = aBall2;
  inputsDataTrig[index2T + 2] = aGoalA2;
  inputsDataTrig[index2T + 3] = aAdv2;
}

/// Add data (save)
/// Parameters:
///  - stearing1
///  - stearing2
void FootChaos::addData(float stearing1, float stearing2) {
  if (tickId >= TICKS_SECOND * GAME_LENGTH)
    throw std::invalid_argument("try adding tick data, but game is to long");

  // Global data
  b2Vec2 ballPos = ball->body->GetPosition();
  data[tickId][0] = scoreTeam1;
  data[tickId][1] = scoreTeam2;
  data[tickId][2] = ballPos.x;
  data[tickId][3] = ballPos.y;

  int n_ = DATA_PER_PLAYER;

  // team 1
  b2Vec2 v1Pos = team1->body->GetPosition();
  float v1Angle = team1->body->GetAngle();
  data[tickId][0 + 4] = v1Pos.x;
  data[tickId][1 + 4] = v1Pos.y;
  data[tickId][2 + 4] = v1Angle;
  data[tickId][3 + 4] = stearing1;
  // team 2
  b2Vec2 v2Pos = team2->body->GetPosition();
  float v2Angle = team2->body->GetAngle();
  data[tickId][0 + 4 + DATA_PER_PLAYER] = v2Pos.x;
  data[tickId][1 + 4 + DATA_PER_PLAYER] = v2Pos.y;
  data[tickId][2 + 4 + DATA_PER_PLAYER] = v2Angle;
  data[tickId][3 + 4 + DATA_PER_PLAYER] = stearing2;
}

void FootChaos::checkIdle() {
  // If already idle, don't change
  if (idle)
    return;

  // Idle game
  bool team1Idle = team1->getOrthogonalSpeed().Normalize() < 1;
  bool team2Idle = team2->getOrthogonalSpeed().Normalize() < 1;
  bool ballIdle = ball->getOrthogonalSpeed().Normalize() < 1;
  bool isIdle = team1Idle && team2Idle && ballIdle;

  if (tickId - idleTick > TICKS_SECOND && isIdle)
    idle = true;
  if (isIdle == false)
    idleTick = tickId;

  // Player far from the ball
  bool team1Far =
      (team1->getPosition() - ball->getPosition()).Normalize() > 200;
  bool team2Far =
      (team2->getPosition() - ball->getPosition()).Normalize() > 200;
  bool isFar = team1Far && team2Far;

  if (tickId - farTick > 10 * TICKS_SECOND && isFar)
    idle = true;
  if (isFar == false)
    farTick = tickId;
}
