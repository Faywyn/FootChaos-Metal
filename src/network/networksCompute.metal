#include <metal_stdlib>
using namespace metal;

float activation(float x) { return 1 / (1 + exp(-x)); }

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
  int gameId = index / (2 * sizeLayer);
  int networkIdAbs = (index - gameId * sizeLayer * 2) / sizeLayer;
  int networkId = (networkIdAbs % 2 == 0) ? network1[gameId] : network2[gameId];
  int depth = index % sizeLayer;

  int weightIndexStart = networkId * nbWeightLayer + depth * sizePreviousLayer;
  int inputIndexStart = (2 * gameId + networkIdAbs) * sizePreviousLayer;

  result[index] = 0;
  for (int i = 0; i < sizePreviousLayer; i++) {
    result[index] += networksWeights[weightIndexStart + i] * inputs[inputIndexStart + i];
  }
  result[index] = activation(result[index]);
}


kernel void networksComputeActivation(device float *inputs,
                                      uint index [[thread_position_in_grid]]) {
                                      return;
  inputs[index] = activation(inputs[index]);
}
