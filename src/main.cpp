#include "config.hpp"
#include "network/network.hpp"
#include "utils/utils.hpp"

#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

int main() {
  srand(time());

  int nbLayer = randomInt(5, 10);
  int *layerSize = (int *)malloc(sizeof(int) * nbLayer);
  for (int i = 0; i < nbLayer; i++) {
    layerSize[i] = randomInt(3, 10);
  }

  Network n = Network();
  n.createRandom(layerSize, nbLayer);
  n.display();

  return 1;
}
