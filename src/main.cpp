#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>

#include "config.hpp"
#include "training/training.hpp"
#include "utils.hpp"

int main(int argc, const char *argv[]) {
  srand(static_cast<unsigned int>(time()));

  int nbNetwork = 500;
  int nbLayer = 8;
  int groupSize = 10;
  int _nbNeuronPerLayer = INPUT_LENGTH * 2;
  int *nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);

  nbNeuronPerLayer[0] = INPUT_LENGTH; // 15
  nbNeuronPerLayer[1] = 20;
  nbNeuronPerLayer[2] = 20;
  nbNeuronPerLayer[3] = 20;
  nbNeuronPerLayer[4] = 16;
  nbNeuronPerLayer[5] = 10;
  nbNeuronPerLayer[6] = 6;
  nbNeuronPerLayer[7] = 2;
  // for (int i = 0; i < nbLayer; i++) {
  //   nbNeuronPerLayer[i] = INPUT_LENGTH * 2;
  // }
  // for (int i = 1; i < 10; i++) {
  //   nbNeuronPerLayer[nbLayer - i] = OUTPUT_LENGTH + i * 3;
  // }
  nbNeuronPerLayer[nbLayer - 1] = OUTPUT_LENGTH;

  // Training *training =
  //     new Training(nbNetwork, groupSize, nbLayer, nbNeuronPerLayer, 2);
  Training *training = new Training(2);

  training->startTraining(10, -1);

  delete training;
  free(nbNeuronPerLayer);

  return 1;
}
