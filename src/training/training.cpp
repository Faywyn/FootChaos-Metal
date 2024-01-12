#include "training.hpp"
#include "../utils.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

Training::Training(int nbNetwork, int groupSize, int nbLayer,
                   int *nbNeuronPerLayer) {

  if (nbNetwork % groupSize != 0)
    throw std::invalid_argument("nbNetwork % groupSize != 0");

  this->nbNetwork = nbNetwork;
  this->nbLayer = nbLayer;
  this->groupSize = groupSize;
  this->nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);
  memcpy(this->nbNeuronPerLayer, nbNeuronPerLayer, sizeof(int) * nbLayer);

  networksManager =
      new NetworksManager(nbNetwork, groupSize, nbLayer, nbNeuronPerLayer);
}

Training::Training(fs::path path) {
  networksManager = new NetworksManager(path);
  this->nbNetwork = networksManager->nbNetwork;
  this->nbLayer = networksManager->nbLayer;
  this->groupSize = networksManager->groupSize;
  this->nbWeightPerNetwork = networksManager->nbWeightPerNetwork;
  this->nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);
  memcpy(this->nbNeuronPerLayer, networksManager->nbNeuronPerLayer, nbLayer);
}

Training::~Training() {
  free(nbNeuronPerLayer);
  delete networksManager;

  return;
  for (int i = 0; i < nbNetwork / groupSize; i++) {
    free(groups[i]);
  }
  free(groups);
  free(nbNetworkPerGroup);
}

void Training::save(fs::path path) { networksManager->saveNetworks(path); }

void Training::threadTraining() {
  int nbGroup = nbNetwork / groupSize;
  std::vector<std::thread> threads;
  threads.reserve(nbGroup);

  for (int i = 0; i < nbGroup; i++) {
    threads.emplace_back(
        [this]() { networksManager->performGeneration(groups); });
  }

  for (std::thread &t : threads) {
    t.join();
  }
}

void Training::createGroups() {
  int nbGroup = nbNetwork / groupSize;
  for (int i = 0; i < nbGroup; i++) {
    nbNetworkPerGroup[i] = 0;
  }

  for (int i = 0; i < nbNetwork; i++) {
    int r;
    do {
      r = randomInt(0, nbGroup - 1);
    } while (nbNetworkPerGroup[r] == groupSize);

    groups[r][nbNetworkPerGroup[r]] = i;
    nbNetworkPerGroup[r] += 1;
  }
}

void Training::performTrain() {
  this->createGroups();
  networksManager->performGeneration(groups);
}

void Training::startTraining(int saveEveryXn, int nbGeneration) {
  int nbGen_ = 0;

  int nbGroup = nbNetwork / groupSize;
  groups = (int **)malloc(sizeof(int *) * nbGroup);
  nbNetworkPerGroup = (int *)malloc(sizeof(int) * nbGroup);
  for (int i = 0; i < nbGroup; i++) {
    groups[i] = (int *)malloc(sizeof(int) * groupSize);
  }

  std::cout << "===== ===== ===== =====\n"
            << "Starting training\n"
            << "===== ===== ===== =====" << std::endl;

  while (nbGen_ < nbGeneration || nbGeneration == -1) {
    performTrain();
  }

  std::cout << "===== ===== ===== =====\n"
            << "Training End\n"
            << "===== ===== ===== =====" << std::endl;
}
