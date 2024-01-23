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
  int nbLayer = 20;
  int groupSize = 10;
  int _nbNeuronPerLayer = INPUT_LENGTH * 2;
  int *nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);

  nbNeuronPerLayer[0] = INPUT_LENGTH; // 15
  nbNeuronPerLayer[1] = 15;
  nbNeuronPerLayer[2] = 15;
  nbNeuronPerLayer[3] = 15;
  nbNeuronPerLayer[4] = 15;
  nbNeuronPerLayer[5] = 15;
  nbNeuronPerLayer[6] = 15;
  nbNeuronPerLayer[7] = 15;
  nbNeuronPerLayer[8] = 15;
  nbNeuronPerLayer[9] = 15;
  nbNeuronPerLayer[10] = 15;
  nbNeuronPerLayer[11] = 15;
  nbNeuronPerLayer[12] = 15;
  nbNeuronPerLayer[13] = 14;
  nbNeuronPerLayer[14] = 12;
  nbNeuronPerLayer[15] = 10;
  nbNeuronPerLayer[16] = 8;
  nbNeuronPerLayer[17] = 6;
  nbNeuronPerLayer[18] = 4;
  nbNeuronPerLayer[nbLayer - 1] = OUTPUT_LENGTH;

  auto creatingTime = time();
  // Training *training =
  //     new Training(nbNetwork, groupSize, nbLayer, nbNeuronPerLayer,
  //                  fs::current_path() / "entrainements/2.txt");
  Training *training = new Training(fs::current_path() / "entrainements/2.txt");

  auto initTime = time();
  training->startTraining(1, -1);

  return 1;
}
