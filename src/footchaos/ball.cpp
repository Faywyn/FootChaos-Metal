#include "../config.hpp"
#include "footchaos.hpp"

#include <box2d/box2d.h>

Ball::Ball(b2World *world) {
  this->world = world;

  // Shape of the ball
  b2CircleShape ballShape;
  ballShape.m_p.Set(0.0f, 0.0f);
  ballShape.m_radius = BALL_RADIUS;

  // Creating ball dynamics
  b2FixtureDef fixtureBallDef;
  fixtureBallDef.shape = &ballShape;
  fixtureBallDef.density = 0.1f;
  fixtureBallDef.friction = 0.1f;

  // Init
  b2BodyDef ballDef;
  ballDef.type = b2_dynamicBody;
  ballDef.position.Set(0, 0);
  body = world->CreateBody(&ballDef);
  body->CreateFixture(&fixtureBallDef);
}

Ball::~Ball() { return; }

b2Vec2 Ball::getOrthogonalSpeed() {
  b2Vec2 ortho = body->GetWorldVector(b2Vec2(0, 1));
  return b2Dot(ortho, body->GetLinearVelocity()) * ortho;
}

b2Vec2 Ball::getNormalSpeed() {
  b2Vec2 normalDroit = body->GetWorldVector(b2Vec2(1, 0));
  return b2Dot(normalDroit, body->GetLinearVelocity()) * normalDroit;
}
b2Vec2 Ball::getPosition() { return body->GetPosition(); }
b2Vec2 Ball::getWorldVector(b2Vec2 vec) { return body->GetWorldVector(vec); }

void Ball::setPosition(float x, float y, float angle) {
  body->SetLinearVelocity(b2Vec2(0, 0));
  body->SetAngularVelocity(0);
  body->SetTransform(b2Vec2(x, y), angle);
}

void Ball::tickFriction() {
  b2Vec2 speed = body->GetLinearVelocity();
  b2Vec2 force = -0.5 * FRICTION_COEF * speed;
  body->ApplyForce(force, body->GetWorldCenter(), true);
};

void Ball::tick() {

  tickFriction();

  b2Vec2 force = b2Vec2(0, 0);
  b2Vec2 pos = body->GetPosition();

  // Apply force close to the wall
  // Right and left wall
  if (abs(pos.y) > GOAL_WIDTH && abs(pos.x) > FIELD_LENGHT * 0.9) {
    float signe = pos.x > 0 ? -1 : 1;
    b2Vec2 murForce = b2Vec2(signe * 10, 0);
    force = force + murForce;
  }

  // Hight and low wall
  if (abs(pos.y) > FIELD_WIDTH * 0.9) {
    float signe = pos.y > 0 ? -1 : 1;
    b2Vec2 murForce = b2Vec2(0, signe * 10);
    force = force + murForce;
  }

  // Diagonal wall
  if (abs(pos.x) > FIELD_LENGHT - LENGTH_ANGLE &&
      abs(pos.y) > FIELD_WIDTH - LENGTH_ANGLE) {
    float x = pos.x - (FIELD_LENGHT * 0.9f - LENGTH_ANGLE);
    float y = FIELD_WIDTH - pos.y;

    if (y < x) {
      float signeX = pos.x > 0 ? -1 : 1;
      float signeY = pos.y > 0 ? -1 : 1;
      b2Vec2 wallForce =
          b2Vec2((float)sqrt(50) * signeX, (float)sqrt(50) * signeY);
      force = force + wallForce;
    }
  }

  body->ApplyForce(force, body->GetWorldCenter(), true);
};
