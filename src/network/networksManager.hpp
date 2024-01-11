#ifndef networksManager_hpp
#define networksManager_hpp

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "../config.hpp"
#include "../footchaos/footchaos.hpp"

class NetworksManager {
private:
  // Weights[i] is a list of all weights from layer i
  // in every network

  int nbNetwork;
  int nbLayer;
  int nbWeightPerNetwork;
  int *nbNeuronPerLayer;
  MTL::Buffer **weights;

  // Games data
  FootChaos **games;
  int nbGame;
  int *networksInGame;

  // Metal
  MTL::ComputePipelineState *activationFunctionPSO;
  MTL::ComputePipelineState *weightFunctionPSO;
  MTL::Device *device;

  // Buffer for metal functions
  MTL::Buffer *bufferResult;
  MTL::Buffer *bufferSizeLayer;
  MTL::Buffer *bufferSizePreviousLayer;

  bool created = false;

public:
  NetworksManager();
  NetworksManager(int nbNetwork, int nbLayer, int *nbNeuronPerLayer);
  ~NetworksManager();

  void createNetworks();
  // void loadNetworks();

  void initGeneration();
  void performTickGeneration();
  void performGeneration();

  float *getScore();

  float *computeWeight(MTL::Buffer *inputs);
  void computeActivation(MTL::ComputeCommandEncoder *computeEncoderA,
                         MTL::CommandBuffer *commandBufferA, int length);
  void initWeightsBuffers(MTL::ComputeCommandEncoder *computeEncoder,
                          int layerIndex, MTL::Buffer *inputs);
  void freeBuffers();
};
#endif /* networksManager_hpp */
