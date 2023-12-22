#include "manager.hpp"

#include <iostream>

Manager::Manager(){};
Manager::~Manager() {
  if (networks != nullptr) {
    for (int i = 0; i < nbNetwork; i++) {
      delete networks[i];
    }
    free(networks);
  }
}

void Manager::create(int nbNetwork, int *sizeOfLayer, int nbLayer) {
  networks = (Network **)malloc(sizeof(Network *) * nbNetwork);
  for (int i = 0; i < nbNetwork; i++) {
    networks[i] = new Network();
    networks[i]->createRandom(sizeOfLayer, nbLayer);
  }
}
