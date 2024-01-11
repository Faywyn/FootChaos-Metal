#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <ctime>
#include <iostream>

#include "config.hpp"
#include "network/networksManager.hpp"
#include "training/training.hpp"

int main(int argc, const char *argv[]) {
  // srand(static_cast<unsigned int>(time()));
  srand(0);

  int nbNetwork = 100;
  int nbLayer = 20;
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

  Training *training = new Training();

  auto creatingTime = time();
  std::cout << "Creating training ..." << std::endl;
  training->create(nbNetwork, nbLayer, nbNeuronPerLayer);
  std::cout << "Training created ! " << time() - creatingTime << "ms"
            << std::endl;

  auto initTime = time();
  std::cout << "Start training ..." << std::endl;
  training->startTraining(0, nbNetwork);
  std::cout << "Trained ! " << time() - initTime << "ms" << std::endl;

  sleep(10);

  return 1;
}
