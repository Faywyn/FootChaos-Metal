#include "utils.hpp"
#include "config.hpp"

#include <iostream>

std::random_device rand_dev;
std::mt19937 generator(rand_dev());
std::uniform_int_distribution<int> distr(0, 10e8);

/// Random float between -1 and 1
float randomFloat() {
  float r = ((float)(distr(generator) * 2) / 10e8f) - 1;
  return r;
}

/// Random int
/// Parameters:
///  - min
///  - max
int randomInt(int min, int max) {
  std::uniform_int_distribution<int> distr(min, max);

  int r = distr(generator);
  return r;
}

/// Gaussiant generated number
/// Parameters:
///  - center
///  - radius
float randomGaussian(float center, float radius) {
  std::normal_distribution<float> distr(center, radius);

  float r = distr(generator);
  return r;
}

/// Get current time
uint64_t time() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

/// Round (ex: round(793.249, 10) = 793.2)
/// Parameters:
///  - x
///  - n
float round(float x, int n) { return (float)(int)(x * n) / (float)n; }

/// Print current stat
/// Parameters:
///  - line (console)
///  - p (percentage done)
///  - sizeBar (loading bar size)
///  - start (time)
///  - current (time)
void printStat(int line, float p, int sizeBar, uint64_t start,
               uint64_t current) {

  float duration = round((float)(time() - start) / 1000, 10);
  float end = round((float)((time() - start) * (1 - p)) / (1000 * p), 10);

  std::cout << "\033[" << line << ";0H" << RESET << "  "
            << "[" << CYAN << std::string(floor(p * sizeBar), '=') << RESET
            << std::string(sizeBar - floor(p * sizeBar), '-') << "] "
            << (int)(p * 100) << "% "
            << "\033[" << line << ";" << sizeBar + 6
            << "H Duration: " << duration << "s "
            << "\033[" << line << ";" << sizeBar + 23 << "H End: " << end
            << "s "
            << "\033[" << line << ";" << sizeBar + 35
            << "H Total: " << duration + end << "s " << std::string(20, ' ')
            << "\033[" << STAT_TAB_START + NB_STATS + 7 << ";0H" << std::endl;
}

/// Print old stat
/// Parameters:
///  - line (console)
///  - id
///  - duration
///  - avrg (score)
///  - avrgPoint (point score)
///  - best (score)
///  - nbMatchPerGen
void printOldStat(int line, int id, uint64_t duration, float avrg,
                  float avrgPoints, float best, int nbMatchPerGen) {

  int off = 0; // Offset
  for (int _id = id; _id != 0; _id /= 10) {
    off += 1;
  }

  std::cout << "\033[" << line << ";0H" << std::string(100, ' ') << std::endl;

  std::cout << "\033[" << line << ";0H"
            << "  " << id
            << " | Duration: " << round((float)duration / 1000, 10) << "s";

  std::cout << "\033[" << line << ";" << 23 + off << "H| Average: " << BLUE
            << round(avrg, 1000) << RESET;

  std::cout << "\033[" << line << ";" << 40 + off
            << "H| Best: " << round(best, 1000);

  std::cout << "\033[" << line << ";" << 53 + off
            << "H | Average Points: " << round(avrgPoints, 1000);

  std::cout << "\033[" << line << ";" << 77 + off
            << "H | Match second: " << GREEN
            << round(1000 * nbMatchPerGen / (float)duration, 10) << RESET
            << std::string(20, ' ') << std::endl;
  ;
}

/// Print global stat
/// Parameters:
///  - duration (global)
///  - min (duration)
///  - max (duration)
void printGlobalStat(float duration, float minD, float maxD) {
  std::cout << "\033[" << STAT_TAB_START + 6 + NB_STATS << ":0H\033[0m"
            << "  "
            << "Global Duration: " << round((float)(duration) / 1000, 100)
            << "s "
            << "Min: " << GREEN << round(minD / 1000, 100) << "s " << RESET
            << "Max: " << RED << round(maxD / 1000, 100) << "s " << RESET
            << std::endl;
}

/// Compare results of 2 networks
/// Parameters:
///  - *a
///  - *b
bool compare(float *a, float *b) {
  return (a[1] > b[1]) || (a[1] == b[1] && a[2] > b[2]);
}

/// In order to shuffle
/// Parameters:
///  - *a
///  - *b
bool compareRdm(float *a, float *b) { return randomFloat() > randomFloat(); }
