#ifndef networksManager_hpp
#define networksManager_hpp

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <filesystem>

#include "../config.hpp"
#include "../footchaos/footchaos.hpp"

namespace fs = std::filesystem;

class NetworksManager {
private:
  // Weights[i] is a list of all weights from layer i
  // in every network
  MTL::Buffer **weights;
  MTL::Buffer *network1;
  MTL::Buffer *network2;
  int **groups;

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
  MTL::Buffer *bufferNbGroup;

  bool created = false;

  void initGeneration();
  void performTickGeneration();

  MTL::Buffer *computeWeight(MTL::Buffer *inputs);
  void computeActivation(MTL::ComputeCommandEncoder *computeEncoderA,
                         MTL::CommandBuffer *commandBufferA, int length);
  void initWeightsBuffers(MTL::ComputeCommandEncoder *computeEncoder,
                          int layerIndex, MTL::Buffer *inputs);
  void freeBuffers();

public:
  int nbNetwork;
  int nbLayer;
  int nbWeightPerNetwork;
  int *nbNeuronPerLayer;
  int groupSize;

  NetworksManager(fs::path path);
  NetworksManager(int nbNetwork, int groupSize, int nbLayer,
                  int *nbNeuronPerLayer);
  void initSystem();
  ~NetworksManager();

  void saveNetworks(fs::path path);

  void performGeneration(int **groups);

  float *getScore();
};
#endif /* networksManager_hpp */
