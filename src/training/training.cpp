#include "training.hpp"
#include "../utils.hpp"

#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

Training::Training() {}

void Training::create(int nbNetwork, int nbLayer, int *nbNeuronPerLayer) {
  this->nbNetwork = nbNetwork;
  // networkModel = new Network(nbLayer, nbNeuronPerLayer);
  // networks = (Network **)malloc(sizeof(Network *) * nbNetwork);
  //
  // for (int i = 0; i < nbNetwork; i++) {
  //   networks[i] = new Network(nbLayer, nbNeuronPerLayer);
  //   networks[i]->randomize();
  // }
  // this->created = true;
}

Training::~Training() {
  if (created == false)
    return;
  // TODO
}

void Training::load(fs::path path) { return; }
void Training::save(fs::path path) { return; }

void Training::threadTraining(int *networkIntGroup, int nbNetworkGroup) {
  // Network **networkGroup =
  //     (Network **)malloc(sizeof(Network *) * nbNetworkGroup);
  // for (int i = 0; i < nbNetwork; i++) {
  //   networkGroup[i] = this->networks[networkIntGroup[i]];
  // }
  //
  // std::cout << "#2.x.3" << std::endl;
  //
  // NetworksManager nManager =
  //     NetworksManager(networkGroup, nbNetworkGroup, networkModel);
  //
  // nManager.performGeneration();
  //
  // std::cout << "#2.x.4" << std::endl;
  //
  // float *rManager = nManager.getScore();
  //
  // std::cout << "nbGroup: " << nbNetworkGroup << std::endl;
  //
  // for (int i = 0; i < nbNetworkGroup; i++) {
  //   this->result[networkIntGroup[i]] = rManager[i];
  // }
  // std::cout << "hfuidzadhza" << std::endl;
}

int **Training::createGroups(int sizeOfGroups) {
  int nbGroup = nbNetwork / sizeOfGroups;

  int **groups = (int **)malloc(sizeof(int *) * nbGroup);
  int *nbNetworkPerGroup = (int *)malloc(sizeof(int) * nbGroup);

  for (int i = 0; i < nbGroup; i++) {
    groups[i] = (int *)malloc(sizeof(int) * sizeOfGroups);
    nbNetworkPerGroup[i] = 0;
  }

  for (int i = 0; i < nbNetwork; i++) {
    int r;
    do {
      r = randomInt(0, nbGroup - 1);
    } while (nbNetworkPerGroup[r] == sizeOfGroups);

    groups[r][nbNetworkPerGroup[r]] = i;
    nbNetworkPerGroup[r] += 1;
  }

  return groups;
}

void Training::performTrain(int sizeOfGroups) {

  NetworksManager n = NetworksManager(nbNetwork, nbLayer, nbNeuronPerLayer);
  n.createNetworks();
  n.initGeneration();
  n.performGeneration();
}

void Training::startTraining(int saveEveryXn, int sizeOfGroups) {
  result = (float *)malloc(sizeof(float) * nbNetwork);
  while (true) {
    std::cout << "Performing a train..." << std::endl;
    performTrain(sizeOfGroups);
    std::cout << "Train done" << std::endl;
    return;
  }
  free(result);
}
