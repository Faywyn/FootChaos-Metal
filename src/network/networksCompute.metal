#include <metal_stdlib>
using namespace metal;

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
  int networkIdAbs = index / sizeLayer;
  int networkId = (networkIdAbs % 2 == 0) ? network1[gameId] : network2[gameId];
  int depth = index % sizeLayer;
    
  int weightIndexStart = networkId * nbWeightLayer + depth * sizePreviousLayer;
  int inputIndexStart = networkIdAbs * sizePreviousLayer;

  result[index] = 0;
  for (int i = 0; i < sizePreviousLayer; i++) {
    result[index] = result[index] + networksWeights[weightIndexStart + i] * inputs[inputIndexStart + i];
  }

  // Pass activation function
  result[index] = 1 / ( 1 + exp(-result[index]));
}

float normalize(float val, float min, float max) {
  return ((val - min) / (max - min)) * 2 - 1;
}


kernel void networksComputeDataNorm(device float *inputs,
                                    device float *result, 
                                    device int *data,
                                    uint index [[thread_position_in_grid]]) {

  int INPUT_NORM_DATA_LENGTH = data[0];
  // int INPUT_TRIG_DATA_LENGTH = data[1];
  int INPUT_LENGTH = data[2];

  int i = index / INPUT_NORM_DATA_LENGTH;
  i = i * INPUT_LENGTH;
  i = i + index % INPUT_NORM_DATA_LENGTH;

  float val = inputs[index * 3 + 0];
  float min = inputs[index * 3 + 1];
  float max = inputs[index * 3 + 2];

  result[i] = ((val - min) / (max - min)) * 2 - 1;
}

kernel void networksComputeDataTrig(device float *inputs,
                                    device float *result,
                                    device int *data,
                                    uint index [[thread_position_in_grid]]) {


  int INPUT_NORM_DATA_LENGTH = data[0];
  int INPUT_TRIG_DATA_LENGTH = data[1];
  int INPUT_LENGTH = data[2];

  int i = index / (INPUT_TRIG_DATA_LENGTH / 2);
  i = i * INPUT_LENGTH + INPUT_NORM_DATA_LENGTH;
  i = i + index % (INPUT_TRIG_DATA_LENGTH / 2);

  float _cos;
  float _sin = sincos(inputs[index] , _cos);

  result[i + 0] = _cos;
  result[i + 1] = _sin;
}
