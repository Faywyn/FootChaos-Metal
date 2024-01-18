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
  MTL::Buffer **groups;

  // Games data
  FootChaos **games;

  // Metal
  MTL::ComputePipelineState *weightFunctionPSO;
  MTL::Device *device;

  void initGeneration();
  void initSystem();
  void initBuffer();
  void performTickGeneration();

  MTL::Buffer *computeNetworks(MTL::Buffer *inputs);

public:
  int nbNetwork;
  int nbLayer;
  int *nbNeuronPerLayer;
  int groupSize;
  int nbGeneration;
  int nbGame;

  NetworksManager(fs::path path);
  NetworksManager(int nbNetwork, int groupSize, int nbLayer,
                  int *nbNeuronPerLayer);
  ~NetworksManager();

  void performGeneration(int **groups);
  float **getScore();

  void saveNetworks(fs::path path);
  void randomizeNetwork(int networkId);
  void copyNetwork(int from, int to);
  void mutateNetwork(int id, float p);
};
#endif /* networksManager_hpp */
