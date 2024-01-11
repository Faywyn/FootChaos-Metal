#include "utils.hpp"
#include <random>

// Between -1 and 1
float randomFloat() {
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_int_distribution<int> distr(0, 10e7);

  float r = ((float)(distr(generator) * 2) / 10e7f) - 1;
  return r;
}

int randomInt(int min, int max) {
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_int_distribution<int> distr(min, max);

  int r = distr(generator);
  return r;
}
