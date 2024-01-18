#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>

#include "config.hpp"
#include "training/training.hpp"
#include "utils.hpp"

int main(int argc, const char *argv[]) {
  srand(static_cast<unsigned int>(time()));

  int nbNetwork = 1000;
  int nbLayer = 20;
  int groupSize = 10;
  int _nbNeuronPerLayer = INPUT_LENGTH;
  int *nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);
  for (int i = 0; i < nbLayer; i++) {
    nbNeuronPerLayer[i] = _nbNeuronPerLayer;
  }
  nbNeuronPerLayer[0] = INPUT_LENGTH;
  nbNeuronPerLayer[nbLayer - 1] = OUTPUT_LENGTH;

  float *inputs = (float *)malloc(sizeof(int) * _nbNeuronPerLayer);
  for (int i = 0; i < _nbNeuronPerLayer; i++) {
    inputs[i] = rand() * (maxWeight - minWeight) + minWeight;
  }

  auto creatingTime = time();
  Training *training =
      new Training(nbNetwork, groupSize, nbLayer, nbNeuronPerLayer,
                   fs::current_path() / "entrainements/1.txt");
  // Training *training = new Training(fs::current_path() /
  // "entrainements/1.txt");

  auto initTime = time();
  training->startTraining(0, 10);

  return 1;
}
