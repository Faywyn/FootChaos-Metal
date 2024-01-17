#include "utils.hpp"

#include <iostream>

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

uint64_t time() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

float round(float x, int n) { return (float)(int)(x * n) / (float)n; }

void printStat(int line, float p, int sizeBar, uint64_t start,
               uint64_t current) {

  std::cout << "\033[" + std::to_string(line) + ";0H"
            << "  "
            << "[" << std::string(floor(p * sizeBar), '=')
            << std::string(sizeBar - floor(p * sizeBar), '-') << "] "
            << (int)(p * 100) << "% "
            << "Duration: " << round((float)(time() - start) / 1000, 10) << "s "
            << "End: "
            << round((float)(time() - start) * (1 - p) / (1000 * p), 10) << "s "
            << std::string(20, ' ') << std::endl;
}

void printOldStat(int line, int id, uint64_t duration) {
  std::cout << "\033[" + std::to_string(line) + ";0H"
            << "  " << id
            << " | Duration: " << round((float)duration / 1000, 10) << "s "
            << std::string(20, ' ') << std::endl;
  ;
}
