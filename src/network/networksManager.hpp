#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <filesystem>
#include <thread>
#include <vector>

#include "../footchaos/footchaos.hpp"
#include "commandManager.hpp"

namespace fs = std::filesystem;

class CommandManager;
class NetworksManager {
private:
  // Groups for the gen
  int **groups;
  bool created = false;

  // Games data
  FootChaos **games;
  fs::path path;

  // Metal
  MTL::Device *device;

  void initGeneration();
  void initSystem();
  void initBuffer();
  void initGames();
  void performTickGeneration(int tickId, std::vector<std::thread> *threads);
  void setInputs(int tickId, std::vector<std::thread> *threads);

  void computeNetworks(int _nbGame, int tickId);
  MTL::CommandBuffer *createCommandBuffer(int _nbGame);

public:
  int nbNetwork;
  int nbLayer;
  int *nbNeuronPerLayer;
  int groupSize;
  int nbGeneration;
  int nbGame;
  int nbWeightPerNetwork;

  MTL::CommandQueue *commandQueue;
  MTL::ComputePipelineState *weightFunctionPSO;
  MTL::ComputePipelineState *dataTrigFunctionPSO;
  MTL::ComputePipelineState *dataNormFunctionPSO;

  // Weights[i] is a list of all weights from layer i
  // in every network
  MTL::Buffer **weights;
  // Data needed for neurone calcul
  MTL::Buffer **data;
  MTL::Buffer *inputData;
  // Result of each layer
  MTL::Buffer ***result;
  // Game info in order de compute all sin, cos ... and normalise
  // inputDataNorm[i + 0]: input
  // inputDataNorm[i + 1]: maxVal
  // inputDataNorm[i + 2]: minVal
  MTL::Buffer *inputDataNorm; // For normalise
  MTL::Buffer *inputDataTrig; // For trigo functions
  // List of network in each game
  MTL::Buffer *network1;
  MTL::Buffer *network2;

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
  void crossover(int id1, int id2, float p)

      void saveGame(int player1, int player2, fs::path path);
  void saveNextGames(bool status, fs::path path);
};
