#include "utils.hpp"
#include <random>

float randomFloat() {
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_int_distribution<int> distr(0, 10e7);

  float r = (float)(distr(generator) / 10e7f);
  return (float)r;
}

float randomFloat(float min, float max) {
  return randomFloat() * (max - min) + min;
}

int randomInt(int min, int max) {
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_int_distribution<int> distr(min, max);

  int r = distr(generator);
  return r;
}

uint64_t time() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}
