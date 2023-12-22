#include "network.hpp"
#include "../utils/utils.hpp"
#include "config.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

Network::Network() {}

Network::~Network() {
  if (weight != nullptr) {
    for (int i = 0; i < nbLayer; i++) {
      free(weight[i]);
    }
    free(weight);
  }
  if (sizeOfLayer != nullptr) {
    free(sizeOfLayer);
  }
  if (sizeOfLayerWeight != nullptr) {
    free(sizeOfLayerWeight);
  }
}

void Network::createRandom(int *sizeOfLayer, int nbLayer) {
  this->nbLayer = nbLayer;
  this->weight = (float **)malloc(sizeof(float *) * (nbLayer));
  this->sizeOfLayer = (int *)malloc(sizeof(int) * nbLayer);
  this->sizeOfLayerWeight = (int *)malloc(sizeof(int) * nbLayer);

  this->weight[0] = nullptr;
  this->sizeOfLayerWeight[0] = 0;

  std::memcpy(this->sizeOfLayer, sizeOfLayer, sizeof(int) * nbLayer);

  for (int i = 1; i < nbLayer; i++) {
    this->sizeOfLayerWeight[i] = sizeOfLayer[i] * sizeOfLayer[i - 1];
  }

  for (int i = 1; i < nbLayer; i++) {
    this->weight[i] = (float *)malloc(sizeof(float) * sizeOfLayerWeight[i]);
    for (int j = 0; j < sizeOfLayerWeight[i]; j++) {
      this->weight[i][j] = randomFloat(MIN_WEIGHT, MAX_WEIGHT);
    }
  }
}

void Network::load(float **weight, int *sizeOfLayer, int nbLayer) {
  this->weight = (float **)malloc(sizeof(float *) * nbLayer);
  this->sizeOfLayer = (int *)malloc(sizeof(int) * nbLayer);
  this->sizeOfLayerWeight = (int *)malloc(sizeof(int) * nbLayer);
  this->nbLayer = nbLayer;

  this->weight[0] = nullptr;
  this->sizeOfLayerWeight[0] = 0;

  std::memcpy(this->sizeOfLayer, sizeOfLayer, sizeof(int) * nbLayer);

  for (int i = 1; i < nbLayer; i++) {
    this->sizeOfLayerWeight[i] = sizeOfLayer[i] * sizeOfLayer[i - 1];
  }

  for (int i = 1; i < nbLayer; i++) {
    this->weight[i] = (float *)malloc(sizeof(float) * sizeOfLayerWeight[i]);

    std::memcpy(this->weight[i], weight[i],
                sizeof(float) * sizeOfLayerWeight[i]);
  }
}

void Network::display() {
  for (int i = 1; i < nbLayer; i++) {
    std::cout << "- Layer: " << i << " - Size of Layer " << i - 1 << ": "
              << sizeOfLayer[i - 1] << " - Size of Layer " << i << ": "
              << sizeOfLayer[i] << std::endl
              << "  ";
    for (int j = 0; j < sizeOfLayerWeight[i]; j++) {
      std::cout << weight[i][j] << " ";
      if ((j + 1) % sizeOfLayer[i - 1] == 0)
        std::cout << std::endl << "  ";
    }
    std::cout << std::endl;
  }
}
