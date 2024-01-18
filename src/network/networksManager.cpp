#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "../utils.hpp"
#include "config.hpp"
#include "networksManager.hpp"

/// Create manager from data
/// Parameters:
///  - path
NetworksManager::NetworksManager(fs::path path) {
  std::ifstream file;
  file.open(path, std::ios::binary);

  // Read first data
  file.read((char *)&nbGeneration, sizeof(int));
  file.read((char *)&nbNetwork, sizeof(int));
  file.read((char *)&nbLayer, sizeof(int));
  file.read((char *)&groupSize, sizeof(int));

  // Load other data (array)
  nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);

  for (int i = 0; i < nbLayer; i++) {
    file.read((char *)&nbNeuronPerLayer[i], sizeof(int));
  }

  this->initSystem();
  this->initBuffer();

  for (int i = 1; i < nbLayer; i++) {
    int nbWeigth = nbNetwork * nbNeuronPerLayer[i] * nbNeuronPerLayer[i - 1];
    float *cont = (float *)weights[i]->contents();
    for (int j = 0; j < nbWeigth; j++) {
      file.read((char *)&cont[j], sizeof(float));
    }
  }
};

/// Create new manager (new networks) from params
/// Parameters:
///  - nbNetwork
///  - groupSize
///  - nbLayer
///  - *nbNeuronPerLayer
NetworksManager::NetworksManager(int nbNetwork, int groupSize, int nbLayer,
                                 int *nbNeuronPerLayer) {
  // Keep datas
  this->nbNetwork = nbNetwork;
  this->groupSize = groupSize;
  this->nbLayer = nbLayer;
  this->nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);
  memcpy(this->nbNeuronPerLayer, nbNeuronPerLayer, sizeof(int) * nbLayer);

  this->initSystem();
  this->initBuffer();

  // Generate random weights
  for (int i = 1; i < nbLayer; i++) {
    int nbWeigth = nbNetwork * nbNeuronPerLayer[i] * nbNeuronPerLayer[i - 1];
    float *values = (float *)weights[i]->contents();
    for (int j = 0; j < nbWeigth; j++) {
      values[j] = randomFloat();
    }
  }
}

/// Create buffers for games and networks
void NetworksManager::initBuffer() {
  this->nbGame = (nbNetwork / groupSize) * (groupSize - 1) * (groupSize) / 2;
  this->network1 =
      device->newBuffer(sizeof(int) * nbGame, MTL::ResourceStorageModeShared);
  this->network2 =
      device->newBuffer(sizeof(int) * nbGame, MTL::ResourceStorageModeShared);

  weights = (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * nbLayer);
  for (int i = 1; i < nbLayer; i++) {
    int nbWeigth = nbNetwork * nbNeuronPerLayer[i] * nbNeuronPerLayer[i - 1];
    weights[i] = device->newBuffer(sizeof(float) * nbWeigth,
                                   MTL::ResourceStorageModeShared);
  }
}

/// Init device, the metals functions, ...
void NetworksManager::initSystem() {
  NS::Error *error = nullptr;

  device = MTL::CreateSystemDefaultDevice();

  // Finding metal functions in the lib
  auto strWeightFunction =
      NS::String::string("networksComputeWeight", NS::ASCIIStringEncoding);

  NS::String *path =
      NS::String::string("shaders.metallib", NS::UTF8StringEncoding);
  MTL::Library *defaultLibrary = device->newLibrary(path, &error);

  MTL::Function *weightFunction =
      defaultLibrary->newFunction(strWeightFunction);
  defaultLibrary->release();

  if (weightFunction == nullptr)
    throw std::invalid_argument("Failed to find the networksComputeWeight");

  // Create a compute pipeline state object.
  weightFunctionPSO = device->newComputePipelineState(weightFunction, &error);
  weightFunction->release();

  if (weightFunctionPSO == nullptr) {
    std::cout << error << std::endl;
    throw std::invalid_argument(
        "Failed to created pipeline (weightFunctionPSO) state object, see "
        "error above.");
  }
}

NetworksManager::~NetworksManager() {
  free(nbNeuronPerLayer);

  if (weightFunctionPSO != nullptr)
    weightFunctionPSO->release();
  if (network1 != nullptr)
    network1->release();
  if (network2 != nullptr)
    network2->release();
}

void NetworksManager::saveNetworks(fs::path path) {
  std::ofstream file;
  file.open(path, std::ios::binary);

  file.write((char *)&nbGeneration, sizeof(int));
  file.write((char *)&nbNetwork, sizeof(int));
  file.write((char *)&nbLayer, sizeof(int));
  file.write((char *)&groupSize, sizeof(int));

  for (int i = 0; i < nbLayer; i++) {
    file.write((char *)&nbNeuronPerLayer[i], sizeof(int));
  }
  for (int i = 1; i < nbLayer; i++) {
    float *cont = (float *)weights[i]->contents();
    int nbWeigth = nbNetwork * nbNeuronPerLayer[i] * nbNeuronPerLayer[i - 1];
    for (int j = 0; j < nbWeigth; j++) {
      file.write((char *)&cont[j], sizeof(float));
    }
  }
  file.close();
}

/// Compute all network with provided inputs
/// Parameters:
///  - *inputs
MTL::Buffer *NetworksManager::computeNetworks(MTL::Buffer *inputs) {

  // Create command queue and buffer
  MTL::CommandQueue *commandQueue = device->newCommandQueue();
  if (commandQueue == nullptr)
    throw "Impossible to create command queue";
  MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
  if (commandBuffer == nullptr)
    throw "Impossible to create command buffer";

  // Data buffer for simple params
  MTL::Buffer *dataBuffer =
      device->newBuffer(sizeof(int) * 2, MTL::ResourceStorageModeShared);
  int *bufferDataCont = (int *)dataBuffer->contents();
  bufferDataCont[0] = 0; // Layer size
  bufferDataCont[1] = 0; // Previous layer size

  // List all buffer and encoder used for each layer (free later)
  MTL::Buffer **buffers =
      (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * (nbLayer - 1));
  MTL::CommandEncoder **encoders = (MTL::CommandEncoder **)malloc(
      sizeof(MTL::CommandEncoder *) * (nbLayer - 1) * 2);

  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
    int sizePreviousLayer = nbNeuronPerLayer[iLayer - 1];
    int sizeLayer = nbNeuronPerLayer[iLayer];

    // Set simple data
    bufferDataCont[0] = sizeLayer;
    bufferDataCont[1] = sizePreviousLayer;

    // Res of a layer
    MTL::Buffer *resBuffer = device->newBuffer(
        sizeof(float) * sizeLayer * nbGame * 2, MTL::ResourceStorageModeShared);
    if (resBuffer == nullptr)
      throw "Impossible to create result buffer (" + std::to_string(iLayer) +
          ")";

    // Nb thread
    int nbExec = nbGame * sizeLayer * 2;
    MTL::Size gridSize = MTL::Size::Make(nbExec, 1, 1);

    NS::UInteger threadGroupSize =
        weightFunctionPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > nbExec) {
      threadGroupSize = nbExec;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Command encoder
    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();
    if (computeEncoder == nullptr)
      throw "Impossible to create compute encoder (" + std::to_string(iLayer) +
          ")";

    // Set function
    computeEncoder->setComputePipelineState(weightFunctionPSO);

    // Set function params'
    computeEncoder->setBuffer(inputs, 0, 0);
    computeEncoder->setBuffer(resBuffer, 0, 1);
    computeEncoder->setBuffer(weights[iLayer], 0, 2);
    computeEncoder->setBuffer(network1, 0, 3);
    computeEncoder->setBuffer(network2, 0, 4);
    computeEncoder->setBuffer(dataBuffer, 0, 5);

    // Encode the command
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    buffers[iLayer - 1] = inputs;
    encoders[(iLayer - 1)] = computeEncoder;

    inputs = resBuffer;
  }

  // Commit the command
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  // Release buffers and encoders
  for (int i = 0; i < nbLayer - 1; i++) {
    buffers[i]->release();
    encoders[i]->release();
  }
  dataBuffer->release();
  commandBuffer->release();
  commandQueue->release();
  free(buffers);
  free(encoders);

  return inputs;
}

/// Create matchs
void NetworksManager::initGeneration() {
  games = (FootChaos **)malloc(sizeof(FootChaos *) * nbGame);

  // Know with network in with match
  int *contNet1 = (int *)network1->contents();
  int *contNet2 = (int *)network2->contents();

  int _nbGame = 0;
  for (int iG = 0; iG < nbNetwork / groupSize; iG++) {
    for (int iP = 0; iP < groupSize; iP++) {
      for (int jP = iP + 1; jP < groupSize; jP++) {
        games[_nbGame] = new FootChaos(2, _nbGame);

        contNet1[_nbGame] = groups[iG][iP];
        contNet2[_nbGame] = groups[iG][jP];

        _nbGame++;
      }
    }
  }

  assert(_nbGame == nbGame);
}

/// Perform a single tick
void NetworksManager::performTickGeneration() {

  // Create buffer inputs
  MTL::Buffer *inputs =
      device->newBuffer(sizeof(float) * INPUT_LENGTH * nbGame * 2,
                        MTL::ResourceStorageModeShared);

  // Dispatch with threads
  std::vector<std::thread> threads1;
  std::vector<std::thread> threads2;

  float *inputsCont = (float *)inputs->contents();
  threads1.reserve(NB_THREAD);

  // Set inputs
  for (int i = 0; i < NB_THREAD; i++) {
    threads1.emplace_back([i, inputsCont, this]() {
      int start = i * nbGame / NB_THREAD;
      int end = (i == NB_THREAD - 1) ? nbGame : (i + 1) * nbGame / NB_THREAD;
      for (int j = start; j < end; j++) {

        float startIndex[2];
        startIndex[0] = (j * 2 + 0) * INPUT_LENGTH;
        startIndex[1] = (j * 2 + 1) * INPUT_LENGTH;

        games[j]->setInputs(inputsCont, startIndex);
      }
    });
  }

  for (std::thread &t : threads1) {
    t.join();
  }

  // Tick with the result
  MTL::Buffer *res = computeNetworks(inputs);
  float *resCont = (float *)res->contents();
  threads2.reserve(NB_THREAD);
  for (int i = 0; i < NB_THREAD; i++) {
    threads2.emplace_back([i, resCont, this]() {
      int start = i * nbGame / NB_THREAD;
      int end = (i == NB_THREAD - 1) ? nbGame : (i + 1) * nbGame / NB_THREAD;
      for (int j = start; j < end; j++) {

        games[j]->tick(resCont);
      }
    });
  }

  for (std::thread &t : threads2) {
    t.join();
  }
  res->release();
}

/// Perform every tick needed for a game
void NetworksManager::performGeneration(int **groups) {

  // Init matchs and groups
  this->groups = groups;
  this->initGeneration();

  // Var for display stats
  int sizeBar = 50;
  int printRate = GAME_LENGTH * TICKS_SECOND / sizeBar;
  auto timeStart = time();

  for (int i = 0; i < GAME_LENGTH * TICKS_SECOND; i++) {

    // Display data
    if (i % printRate == 0 || i == (GAME_LENGTH * TICKS_SECOND - 1)) {
      float p = (float)i / (float)(GAME_LENGTH * TICKS_SECOND - 1);
      printStat(STAT_TAB_START + 1, p, sizeBar, timeStart, time());
    }

    // Perform tick
    performTickGeneration();
  }
}

/// Get the result of gen
float **NetworksManager::getScore() {
  float **res = (float **)malloc(sizeof(float *) * nbNetwork);

  // res[i][0]: network id
  // res[i][1]: score
  for (int i = 0; i < nbNetwork; i++) {
    res[i] = (float *)malloc(sizeof(float) * 2);
    res[i][0] = i;
    res[i][1] = 0;
  }

  // Sum each score
  int _nbGame = 0;
  for (int iG = 0; iG < nbNetwork / groupSize; iG++) {
    for (int iP = 0; iP < groupSize; iP++) {
      for (int jP = iP + 1; jP < groupSize; jP++) {
        res[groups[iG][iP]][1] += games[_nbGame]->scoreTeam1;
        res[groups[iG][jP]][1] += games[_nbGame]->scoreTeam2;

        _nbGame++;
      }
    }
  }

  return res;
}

/// Copy network to another
/// Parameters:
///  - from
///  - to
void NetworksManager::copyNetwork(int from, int to) {}

/// Randomize a network
/// Parameters:
///  - id
void NetworksManager::randomizeNetwork(int id) {}

/// Mutate network
/// Parameters:
///  - id
///  - p: proba to change a weight
void NetworksManager::mutateNetwork(int id, float p) {}
