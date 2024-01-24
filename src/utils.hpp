#include <ctime>
#include <random>

#include "config.hpp"

float randomFloat();
int randomInt(int min, int max);
float round(float x, int n);

uint64_t time();

void printStat(int line, float p, int sizeBar, uint64_t start,
               uint64_t current);
void printOldStat(int line, int id, uint64_t duration, float avrg,
                  float avrgPoints, float best, int nbMatchPerGen);
void printGlobalStat(float duration, float minD, float maxD);

bool compare(float *a, float *b);
bool compareRdm(float *a, float *b);

void mutliThread(void (*func)());
