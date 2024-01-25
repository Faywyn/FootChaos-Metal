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
  bool created = false;

  MTL::CommandQueue *commandQueue;

  // Games data
  FootChaos **games;
  fs::path path;

  // Metal
  MTL::ComputePipelineState *weightFunctionPSO;
  MTL::Device *device;

  void initGeneration();
  void initSystem();
  void initBuffer();
  void initGames();
  void performTickGeneration();

  MTL::Buffer *computeNetworks(MTL::Buffer *inputs, int nbGame);

public:
  int nbNetwork;
  int nbLayer;
  int *nbNeuronPerLayer;
  int groupSize;
  int nbGeneration;
  int nbGame;
  int nbWeightPerNetwork;

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

  void saveGame(int player1, int player2, fs::path path);
  void saveNextGames(bool status, fs::path path);
};
#endif /* networksManager_hpp */
