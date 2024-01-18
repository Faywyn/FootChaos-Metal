#include "utils.hpp"
#include "config.hpp"

#include <iostream>
#include <thread>
#include <vector>

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
            << std::string(20, ' ') << "\033[" << STAT_TAB_START + NB_STATS + 7
            << ";0H" << std::endl;
}

void printOldStat(int line, int id, uint64_t duration) {
  std::cout << "\033[" + std::to_string(line) + ";0H"
            << "  " << id
            << " | Duration: " << round((float)duration / 1000, 10) << "s "
            << std::string(20, ' ') << std::endl;
  ;
}

bool compare(float *a, float *b) {
  return (a[1] > b[1]) || (a[1] == b[1] && a[2] > b[2]);
}

bool compareRdm(float *a, float *b) { return randomFloat() > randomFloat(); }

void mutliThread(void (*func)()) {
  std::vector<std::thread> threads;
  threads.reserve(NB_THREAD);

  for (int i = 0; i < NB_THREAD; i++) {
    threads.emplace_back([func]() { func(); });
  }

  for (std::thread &t : threads) {
    t.join();
  }
}
