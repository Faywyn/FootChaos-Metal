#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>

#include "config.hpp"
#include "training/training.hpp"
#include "utils.hpp"

int main(int argc, const char *argv[]) {
  srand(static_cast<unsigned int>(time()));

  if (argc < 3 || (std::stoi(argv[2]) == 1) && argc < 5) {
    std::cout
        << "./FootChaos id_training 0 (run existing training)\n"
        << "./FootChaos id_training 1 nbNetwork groupSize (create training)"
        << std::endl;
    throw std::invalid_argument("Invalid command argument, see message above");
  }

  int id = std::stoi(argv[1]);
  bool create = std::stoi(argv[2]) == 1;

  if (create) {
    int nbNetwork = std::stoi(argv[3]);
    int groupSize = std::stoi(argv[4]);

    Training training = Training(nbNetwork, groupSize, NB_LAYER,
                                 (int *)NB_NEURON_PER_LAYER, id);
    training.save();
  }

  Training *training = new Training(id);
  training->startTraining(10, -1);
  return 1;
}
