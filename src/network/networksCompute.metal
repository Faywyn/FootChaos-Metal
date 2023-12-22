#include <metal_stdlib>
using namespace metal;


// networksWeights: List of all the weights of every networks
// inputs: Result list from all the previous layer

kernel void  networksComputeWeight(device float* inputs,
                                   device float* result,
                                   device float* networksWeights,
                                   device int* _sizeLayer,
                                   device int* _sizePreviousLayer,
                                   uint index [[thread_position_in_grid]]) {
    
    // Get the values
    int sizeLayer = *_sizeLayer;
    int sizePreviousLayer = *_sizePreviousLayer;
    int nbWeightLayer = sizeLayer * sizePreviousLayer;
    
    // Determine index of Network, Neuron and Weight
    int indexNetwork = index / nbWeightLayer;
    int indexInput = (index % sizePreviousLayer) + sizePreviousLayer * indexNetwork;
    int indexOutput = (index / sizePreviousLayer)  ;
    
    result[indexOutput] += inputs[indexInput] * networksWeights[index];
}

float activation(float x) {
    return 1 / (1 + exp(-x));
}

kernel void  networksComputeActivation(device float* inputs, uint index [[thread_position_in_grid]]) {
    inputs[index] = activation(inputs[index]);
}

