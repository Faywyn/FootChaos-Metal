#include <ctime>
#include <random>

float randomFloat();
int randomInt(int min, int max);
uint64_t time();
float round(float x, int n);
void printStat(int line, float p, int sizeBar, uint64_t start,
               uint64_t current);
void printOldStat(int line, int id, uint64_t duration);
