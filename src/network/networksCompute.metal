#include <metal_stdlib>
using namespace metal;

// networksWeights: List of all the weights of every networks
// inputs: Result list from all the previous layer

kernel void networksComputeWeight(device float *inputs,
                                  device float *result,
                                  device float *networksWeights,
                                  device int *network1,
                                  device int *network2,
                                  device int *data,
                                  uint index [[thread_position_in_grid]]) {

  // Get the values
  int sizeLayer = data[0]; 
  int sizePreviousLayer = data[1];
  int nbWeightLayer = sizeLayer * sizePreviousLayer;

  // Determine index of Network, Neuron, Weight, ...
  int gameId = index / (2 * nbWeightLayer);
  int networkIdAbs = (index - gameId * nbWeightLayer * 2) / nbWeightLayer;
  int networkId = (networkIdAbs % 2 == 0) ? network1[gameId] : network2[gameId];

  int indexInput = gameId * sizePreviousLayer + (index % sizeLayer);
  int indexOutput = gameId * sizeLayer + index / sizeLayer;
  int indexWeight = networkId * nbWeightLayer + index % nbWeightLayer;

  result[indexOutput] += inputs[indexInput] * networksWeights[indexWeight];
}

float activation(float x) { return 1 / (1 + exp(-x)); }

kernel void networksComputeActivation(device float *inputs,
                                      uint index [[thread_position_in_grid]]) {
  inputs[index] = activation(inputs[index]);
}
