#include <cmath>
#include <cstdlib>
#include <cstring>
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

NetworksManager::NetworksManager(fs::path path) {
  this->initSystem();

  throw "Feature not implemented";
};

NetworksManager::NetworksManager(int nbNetwork, int groupSize, int nbLayer,
                                 int *nbNeuronPerLayer) {
  this->nbNetwork = nbNetwork;
  this->groupSize = groupSize;
  this->nbGame = (nbNetwork / groupSize) * (groupSize - 1) * (groupSize) / 2;
  this->nbLayer = nbLayer;
  this->nbWeightPerNetwork = 0;
  this->nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);
  memcpy(this->nbNeuronPerLayer, nbNeuronPerLayer, sizeof(int) * nbLayer);

  this->initSystem();

  this->network1 = device->newBuffer(sizeof(int) * nbNetwork * groupSize,
                                     MTL::ResourceStorageModeShared);
  this->network2 = device->newBuffer(sizeof(int) * nbNetwork * groupSize,
                                     MTL::ResourceStorageModeShared);
  this->bufferNbGroup =
      device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);

  int *nbGroupContent = (int *)bufferNbGroup->contents();
  nbGroupContent[0] = nbNetwork / groupSize;

  for (int i = 1; i < nbLayer; i++) {
    nbWeightPerNetwork += nbNeuronPerLayer[i] * nbNeuronPerLayer[i - 1];
  }

  weights = (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * nbLayer);
  for (int i = 1; i < nbLayer; i++) {
    weights[i] = device->newBuffer(sizeof(float) * nbNeuronPerLayer[i - 1] *
                                       nbNeuronPerLayer[i],
                                   MTL::ResourceStorageModeShared);

    float *values = (float *)weights[i]->contents();
    for (int j = 0; j < nbNeuronPerLayer[i - 1] * nbNeuronPerLayer[i]; j++) {
      values[j] = randomFloat();
    }
  }
}

void NetworksManager::initSystem() {
  NS::Error *error = nullptr;

  device = MTL::CreateSystemDefaultDevice();

  // Finding metal functions in the lib
  auto strWeightFunction =
      NS::String::string("networksComputeWeight", NS::ASCIIStringEncoding);
  auto strActivationFunction =
      NS::String::string("networksComputeActivation", NS::ASCIIStringEncoding);

  NS::String *path =
      NS::String::string("shaders.metallib", NS::UTF8StringEncoding);
  MTL::Library *defaultLibrary = device->newLibrary(path, &error);

  MTL::Function *weightFunction =
      defaultLibrary->newFunction(strWeightFunction);
  MTL::Function *activationFunction =
      defaultLibrary->newFunction(strActivationFunction);
  defaultLibrary->release();

  if (weightFunction == nullptr)
    throw std::invalid_argument("Failed to find the networksComputeWeight");
  if (activationFunction == nullptr)
    throw std::invalid_argument("Failed to find the networksComputeActivation");

  // Create a compute pipeline state object.
  weightFunctionPSO = device->newComputePipelineState(weightFunction, &error);
  activationFunctionPSO =
      device->newComputePipelineState(activationFunction, &error);
  weightFunction->release();
  activationFunction->release();

  if (weightFunctionPSO == nullptr) {
    std::cout << error << std::endl;
    throw std::invalid_argument(
        "Failed to created pipeline (weightFunctionPSO) state object, see "
        "error above.");
  }
  if (activationFunctionPSO == nullptr) {
    std::cout << error << std::endl;
    throw std::invalid_argument(
        "Failed to created pipeline (activationFunctionPSO) state object, see "
        "error above.");
  }
}

NetworksManager::~NetworksManager() {
  free(nbNeuronPerLayer);

  if (weightFunctionPSO != nullptr)
    weightFunctionPSO->release();
  if (activationFunctionPSO != nullptr)
    activationFunctionPSO->release();
  if (network1 != nullptr)
    network1->release();
  if (network2 != nullptr)
    network2->release();
}

void NetworksManager::saveNetworks(fs::path path) {
  throw "Feature not implemented";
}

void NetworksManager::initWeightsBuffers(
    MTL::ComputeCommandEncoder *computeEncoder, int layerIndex,
    MTL::Buffer *inputs) {

  int sizePreviousLayer = nbNeuronPerLayer[layerIndex - 1];
  int sizeLayer = nbNeuronPerLayer[layerIndex];

  bufferResult = device->newBuffer(sizeof(float) * nbGame * 2 * sizeLayer,
                                   MTL::ResourceStorageModeShared);

  float *resultsContent = (float *)bufferResult->contents();
  int *sizeLayerContent = (int *)bufferSizeLayer->contents();
  int *sizePreviousLayerContent = (int *)bufferSizePreviousLayer->contents();

  assert(resultsContent);
  assert(sizeLayerContent);
  assert(sizePreviousLayerContent);

  for (int i = 0; i < nbGame * 2 * sizeLayer; i++) {
    resultsContent[i] = 0;
  }

  sizeLayerContent[0] = sizeLayer;
  sizePreviousLayerContent[0] = sizePreviousLayer;

  computeEncoder->setBuffer(inputs, 0, 0);
  computeEncoder->setBuffer(bufferResult, 0, 1);
  computeEncoder->setBuffer(weights[layerIndex], 0, 2);
  computeEncoder->setBuffer(network1, 0, 3);
  computeEncoder->setBuffer(network2, 0, 4);
  computeEncoder->setBuffer(bufferSizeLayer, 0, 5);
  computeEncoder->setBuffer(bufferSizePreviousLayer, 0, 6);
  computeEncoder->setBuffer(bufferNbGroup, 0, 7);
}

MTL::Buffer *NetworksManager::computeNetworks(MTL::Buffer *inputs) {
  MTL::CommandQueue *commandQueue = device->newCommandQueue();
  if (commandQueue == nullptr)
    throw "Impossible to create command queue";
  MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
  if (commandBuffer == nullptr)
    throw "Impossible to create command buffer";

  MTL::Buffer *dataBuffer =
      device->newBuffer(sizeof(int) * 2, MTL::ResourceStorageModeShared);
  int *bufferDataCont = (int *)dataBuffer->contents();
  bufferDataCont[0] = 0; // Layer size
  bufferDataCont[1] = 0; // Previous layer size

  MTL::Buffer **buffers =
      (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * (nbLayer - 1));
  MTL::CommandEncoder **encoders = (MTL::CommandEncoder **)malloc(
      sizeof(MTL::CommandEncoder *) * (nbLayer - 1) * 2);

  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
    int sizePreviousLayer = nbNeuronPerLayer[iLayer - 1];
    int sizeLayer = nbNeuronPerLayer[iLayer];
    bufferDataCont[0] = sizeLayer;
    bufferDataCont[1] = sizePreviousLayer;

    MTL::Buffer *resBuffer = device->newBuffer(
        sizeof(float) * sizeLayer * nbGame * 2, MTL::ResourceStorageModeShared);
    if (resBuffer == nullptr)
      throw "Impossible to create result buffer (" + std::to_string(iLayer) +
          ")";

    int nbExec = nbGame * sizeLayer * 2;
    MTL::Size gridSize = MTL::Size::Make(nbExec, 1, 1);

    NS::UInteger threadGroupSize =
        weightFunctionPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > nbExec) {
      threadGroupSize = nbExec;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();
    if (computeEncoder == nullptr)
      throw "Impossible to create compute encoder (" + std::to_string(iLayer) +
          ")";

    computeEncoder->setComputePipelineState(weightFunctionPSO);

    computeEncoder->setBuffer(inputs, 0, 0);
    computeEncoder->setBuffer(bufferResult, 0, 1);
    computeEncoder->setBuffer(weights[iLayer], 0, 2);
    computeEncoder->setBuffer(network1, 0, 3);
    computeEncoder->setBuffer(network2, 0, 4);
    computeEncoder->setBuffer(dataBuffer, 0, 5);

    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    buffers[iLayer - 1] = inputs;
    encoders[(iLayer - 1)] = computeEncoder;

    inputs = resBuffer;
  }

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  commandBuffer->release();

  for (int i = 0; i < nbLayer - 1; i++) {
    buffers[i]->release();
    encoders[i]->release();
  }
  dataBuffer->release();
  commandQueue->release();
  free(buffers);
  free(encoders);

  return inputs;
}

void NetworksManager::initGeneration() {
  games = (FootChaos **)malloc(sizeof(FootChaos *) * nbGame);
  networksInGame = (int *)malloc(sizeof(int) * nbGame * 2);

  assert(games);
  assert(networksInGame);

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

void NetworksManager::performTickGeneration() {
  MTL::Buffer *inputs =
      device->newBuffer(sizeof(float) * INPUT_LENGTH * nbGame * 2,
                        MTL::ResourceStorageModeShared);
  std::vector<std::thread> threads1;
  std::vector<std::thread> threads2;

  float *inputsCont = (float *)inputs->contents();
  threads1.reserve(NB_THREAD);

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

  // -

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

void NetworksManager::performGeneration(int **groups) {
  this->groups = groups;
  this->initGeneration();

  std::cout << "\033[0;0H"
            << "Nb Group: " << nbNetwork / groupSize
            << "\nNb Network: " << nbNetwork << "\nNb Game: " << nbGame
            << std::endl;

  int sizeBar = 50;
  int printRate = GAME_LENGTH * TICKS_SECOND / sizeBar;
  auto timeStart = time();

  for (int i = 0; i < GAME_LENGTH * TICKS_SECOND; i++) {
    if (i % printRate == 0 || i == (GAME_LENGTH * TICKS_SECOND - 1)) {
      float p = (float)i / (float)(GAME_LENGTH * TICKS_SECOND - 1);
      printStat(STAT_TAB_START + 1, p, sizeBar, timeStart, time());
    }
    performTickGeneration();
  }
}

float *NetworksManager::getScore() {
  float *res = (float *)malloc(sizeof(float) * nbNetwork);

  for (int i = 0; i < nbNetwork; i++) {
    res[i] = 0;
  }

  int _nbMatch = 0;
  for (int i = 0; i < nbNetwork; i++) {
    for (int j = i + 1; j < nbNetwork; j++) {
      FootChaos *match = games[_nbMatch];

      res[i] += match->scoreTeam1;
      res[j] += match->scoreTeam2;

      _nbMatch++;
    }
  }

  return res;
}
