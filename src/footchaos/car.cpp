#include "../config.hpp"
#include "footchaos.hpp"

#include <box2d/box2d.h>
#include <iostream>
#include <stdlib.h>

/// Car class
/// Parameters:
///  - world
Car::Car(b2World *world) {
  this->world = world;

  // Dynamic of the car
  b2BodyDef carDef;
  carDef.type = b2_dynamicBody;
  carDef.position.Set(0, 0);
  carDef.angle = 0;
  body = world->CreateBody(&carDef);

  // Shape
  b2PolygonShape polyShape;
  polyShape.SetAsBox(CAR_WIDTH / 2, CAR_LENGHT / 2);

  b2FixtureDef fixtureDef;
  fixtureDef.shape = &polyShape;
  fixtureDef.density = 1.0f;
  fixtureDef.friction = 0.3f;

  body->CreateFixture(&fixtureDef);
}

/// Car destructor
Car::~Car() { return; }

/// Get steering
float Car::getSteering() { return steering; }
/// Get position
b2Vec2 Car::getPosition() { return body->GetPosition(); }
/// Get world vector
b2Vec2 Car::getWorldVector(b2Vec2 vec) { return body->GetWorldVector(vec); }

/// Get orthogonal speed
b2Vec2 Car::getOrthogonalSpeed() {
  b2Vec2 ortho = body->GetWorldVector(b2Vec2(0, 1));
  return b2Dot(ortho, body->GetLinearVelocity()) * ortho;
}

/// Get normal speed
b2Vec2 Car::getNormalSpeed() {
  b2Vec2 normal = body->GetWorldVector(b2Vec2(1, 0));
  return b2Dot(normal, body->GetLinearVelocity()) * normal;
}

/// Car set posisition
/// Parameters:
///  - x
///  - y
///  - angle
void Car::setPosition(float x, float y, float angle) {
  body->SetLinearVelocity(b2Vec2(0, 0));
  body->SetAngularVelocity(0);
  body->SetTransform(b2Vec2(x, y), angle);
}

/// Add friction to the car
void Car::tickFriction() {
  // Normal speed
  b2Vec2 impulsion = body->GetMass() * (-getNormalSpeed());
  if (impulsion.Length() > MAX_LATERAL_IMPULSE)
    impulsion *= MAX_LATERAL_IMPULSE / impulsion.Length();
  body->ApplyLinearImpulse(impulsion, body->GetWorldCenter(), false);

  // Angle
  body->ApplyAngularImpulse(
      0.1f * body->GetInertia() * -body->GetAngularVelocity(), false);

  // Ortho speed
  b2Vec2 orthSpeed = getOrthogonalSpeed();
  float speed = orthSpeed.Normalize();
  float force = -FRICTION_COEF * speed * speed;
  float aForce = abs(force);
  float minForce = 500;
  if (aForce > 0 && aForce < minForce) {
    force *= minForce / aForce;
  }
  body->ApplyForce(force * orthSpeed, body->GetWorldCenter(), false);
}

/// Perform a tick on the car
/// Parameters:
///  - speedControler (between 0 and 1)
///  - steeringControler (between 0 and 1)
void Car::tick(float speedControler, float steeringControler) {
  tickFriction();

  // Inputs beetween -1 and +1
  speedControler = speedControler * 2 - 1;
  steeringControler = steeringControler * 2 - 1;

  if (abs(steeringControler - steering) > MAX_STEERING_SPEED) {
    steering += (steeringControler - steering) > 0 ? +MAX_STEERING_SPEED
                                                   : -MAX_STEERING_SPEED;
  } else {
    steering = steeringControler;
  }
  steering = steering > +1.0 ? +1.0 : steering;
  steering = steering < -1.0 ? -1.0 : steering;

  // Modify speed
  b2Vec2 orthospeed = body->GetWorldVector(b2Vec2(0, 1));
  float speed = b2Dot(getOrthogonalSpeed(), orthospeed);
  float force = MAX_FORCE * speedControler;
  if (speedControler < 0 && speed < 0) {
    force *= 0.2;
  }
  body->ApplyForce(force * orthospeed, body->GetWorldCenter(), true);

  // Modify angle
  float steeringControlerMvmt = steering * (speed / MAX_SPEED);
  float desireTorque = steeringControlerMvmt * (float)TORQUE_FORCE;

  body->ApplyTorque(desireTorque, false);
}
