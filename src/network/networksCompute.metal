#include <metal_stdlib>
using namespace metal;

// networksWeights: List of all the weights of every networks
// inputs: Result list from all the previous layer

kernel void networksComputeWeight(device float *inputs, device float *result,
                                  device float *networksWeights,
                                  device int *network1, device int *network2,
                                  device int *_sizeLayer,
                                  device int *_sizePreviousLayer,
                                  device int *_nbGame,
                                  uint index [[thread_position_in_grid]]) {

  // Get the values
  int sizeLayer = *_sizeLayer;
  int sizePreviousLayer = *_sizePreviousLayer;
  int nbGame = *_nbGame;
  int nbWeightLayer = sizeLayer * sizePreviousLayer;

  // Determine index of Network, Neuron, Weight, ...
  int gameId2 = index / (nbWeightLayer * nbGame);
  int gameId = gameId2 / 2;
  int networkId = (gameId2 % 2 == 0) ? network1[gameId] : network2[gameId];

  int indexInput = gameId2 * sizePreviousLayer + (index % sizePreviousLayer);
  int indexOutput = gameId2 * sizeLayer + index / sizeLayer;
  int indexWeight = networkId * nbWeightLayer + index % nbWeightLayer;

  result[indexOutput] += inputs[indexInput] * networksWeights[indexWeight];
}

float activation(float x) { return 1 / (1 + exp(-x)); }

kernel void networksComputeActivation(device float *inputs,
                                      uint index [[thread_position_in_grid]]) {
  inputs[index] = activation(inputs[index]);
}
