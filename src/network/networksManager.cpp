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
  throw "Feature not implemented";
};

NetworksManager::NetworksManager(int nbNetwork, int groupSize, int nbLayer,
                                 int *nbNeuronPerLayer) {
  this->nbNetwork = nbNetwork;
  this->groupSize = groupSize;
  this->nbGamePerGroup = (groupSize - 1) * groupSize / 2;
  this->nbGame = (nbNetwork / groupSize) * nbGamePerGroup;
  this->nbLayer = nbLayer;
  this->nbWeightPerNetwork = 0;
  this->nbNeuronPerLayer = (int *)malloc(sizeof(int) * nbLayer);
  memcpy(this->nbNeuronPerLayer, nbNeuronPerLayer, sizeof(int) * nbLayer);

  this->initSystem();

  this->network1 = device->newBuffer(sizeof(int) * nbNetwork * groupSize,
                                     MTL::ResourceStorageModeShared);
  this->network2 = device->newBuffer(sizeof(int) * nbNetwork * groupSize,
                                     MTL::ResourceStorageModeShared);

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
  std::cout << "dest" << std::endl;
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
    MTL::Buffer *inputs) {}

void NetworksManager::freeBuffers() { return; }

void NetworksManager::computeActivation(
    MTL::ComputeCommandEncoder *computeEncoderA,
    MTL::CommandBuffer *commandBufferA, int length) {

  MTL::Size gridSize = MTL::Size::Make(length * nbGamePerGroup * 2, 1, 1);

  // Calculate a threadgroup size.
  NS::UInteger threadGroupSize =
      weightFunctionPSO->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > length * nbGamePerGroup * 2) {
    threadGroupSize = length * nbGamePerGroup * 2;
  }
  MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

  computeEncoderA->dispatchThreads(
      gridSize, threadgroupSize); // Encode the compute command.
  computeEncoderA->endEncoding(); // End the compute pass.
  commandBufferA->commit();       // Execute the command.
  commandBufferA->waitUntilCompleted();
}

float *NetworksManager::computeWeight(MTL::Buffer *inputs, int groupId) {
  MTL::CommandQueue *commandeQueueW = device->newCommandQueue();
  MTL::CommandQueue *commandeQueueA = device->newCommandQueue();

  if (commandeQueueW == nullptr || commandeQueueA == nullptr)
    throw std::invalid_argument("Failed to find the command queue");

  MTL::Buffer *bufferData =
      device->newBuffer(sizeof(int) * 4, MTL::ResourceStorageModeShared);

  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
    MTL::Buffer *bufferResult = device->newBuffer(
        sizeof(float) * nbGamePerGroup * 2 * nbNeuronPerLayer[1],
        MTL::ResourceStorageModeShared);

    // Create a command buffer to hold commands.
    MTL::CommandBuffer *commandBufferW = commandeQueueW->commandBuffer();
    MTL::CommandBuffer *commandBufferA = commandeQueueA->commandBuffer();
    if (commandBufferW == nullptr || commandBufferA == nullptr)
      throw std::invalid_argument("Failed to init commandBuffer");

    // Start a compute pass.
    MTL::ComputeCommandEncoder *computeEncoderW =
        commandBufferW->computeCommandEncoder();
    MTL::ComputeCommandEncoder *computeEncoderA =
        commandBufferA->computeCommandEncoder();
    if (computeEncoderW == nullptr || computeEncoderA == nullptr)
      throw std::invalid_argument("Failed to init commandEncoder");

    // Encode the pipeline state object and its parameters.
    computeEncoderW->setComputePipelineState(weightFunctionPSO);
    computeEncoderA->setComputePipelineState(activationFunctionPSO);

    int sizePreviousLayer = nbNeuronPerLayer[iLayer - 1];
    int sizeLayer = nbNeuronPerLayer[iLayer];

    float *resultsContent = (float *)bufferResult->contents();

    for (int i = 0; i < nbGamePerGroup * 2 * sizeLayer; i++) {
      resultsContent[i] = 0;
    }

    int *dataCont = (int *)bufferData->contents();
    dataCont[0] = sizeLayer;
    dataCont[1] = sizePreviousLayer;
    dataCont[2] = nbGamePerGroup;
    dataCont[3] = groupId;

    computeEncoderW->setBuffer(inputs, 0, 0);
    computeEncoderW->setBuffer(bufferResult, 0, 1);
    computeEncoderW->setBuffer(weights[iLayer], 0, 2);
    computeEncoderW->setBuffer(network1, 0, 3);
    computeEncoderW->setBuffer(network2, 0, 4);
    computeEncoderW->setBuffer(bufferData, 0, 5);

    // -----
    // Compute the layer values
    // -----

    int nbExec = nbGamePerGroup * sizeLayer * sizePreviousLayer * 2;
    MTL::Size gridSize = MTL::Size::Make(nbExec, 1, 1);

    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize =
        weightFunctionPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > nbExec) {
      threadGroupSize = nbExec;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    computeEncoderW->dispatchThreads(
        gridSize, threadgroupSize); // Encode the compute command.
    computeEncoderW->endEncoding(); // End the compute pass.
    commandBufferW->commit();       // Execute the command.
    commandBufferW->waitUntilCompleted();

    commandBufferW->release();
    computeEncoderW->release();

    // -----
    // Pass the activation function
    // -----

    computeEncoderA->setBuffer(bufferResult, 0, 0);
    computeActivation(computeEncoderA, commandBufferA, sizeLayer);

    commandBufferA->release();
    computeEncoderA->release();

    inputs->release();
    inputs = bufferResult;
  }

  commandeQueueA->release();
  commandeQueueW->release();

  bufferData->release();

  return (float *)inputs->contents();
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
    int *groupCont = (int *)groups[iG]->contents();
    for (int iP = 0; iP < groupSize; iP++) {
      for (int jP = iP + 1; jP < groupSize; jP++) {
        games[_nbGame] = new FootChaos(2, _nbGame);

        contNet1[_nbGame] = groupCont[iP];
        contNet2[_nbGame] = groupCont[jP];

        _nbGame++;
      }
    }
  }
  assert(_nbGame == nbGame);
}

void NetworksManager::performTickGeneration(int groupId) {
  MTL::Buffer *inputs =
      device->newBuffer(sizeof(float) * INPUT_LENGTH * nbGamePerGroup * 2,
                        MTL::ResourceStorageModeShared);

  float *inputsCont = (float *)inputs->contents();

  for (int i = 0; i < nbGamePerGroup; i++) {
    float *startIndex = (float *)malloc(sizeof(float) * 2);
    startIndex[0] = (i * 2 + 0) * INPUT_LENGTH;
    startIndex[1] = (i * 2 + 1) * INPUT_LENGTH;

    games[groupId * nbGamePerGroup + i]->setInputs(inputsCont, startIndex);
    free(startIndex);
  }

  float *res = computeWeight(inputs, groupId);
  for (int i = 0; i < nbGamePerGroup; i++) {
    games[groupId * nbGamePerGroup + i]->tick(res);
  };
}

void NetworksManager::performGenerationGroup(int groupId) {
  int nbTick = GAME_LENGTH * TICKS_SECOND;
  for (int i = 0; i < nbTick; i++) {
    performTickGeneration(groupId);
    if ((i + 1) % (nbTick / 10) == 0) {
      std::cout << i + 1 << "/" << nbTick << std::endl;
    }
  }
}

void NetworksManager::performGeneration(int **groups) {
  int nbGroup = nbNetwork / groupSize;

  this->groups = (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * nbGroup);
  for (int i = 0; i < nbGroup; i++) {
    this->groups[i] = device->newBuffer(sizeof(int) * groupSize,
                                        MTL::ResourceStorageModeShared);
    memcpy(this->groups[i]->contents(), groups[i], sizeof(int) * groupSize);
  }
  this->initGeneration();

  std::cout << "Nb Group: " << nbNetwork / groupSize
            << "\nNb Network: " << nbNetwork << "\nNb Game: " << nbGame
            << std::endl;

  std::vector<std::thread> threads;
  threads.reserve(nbGroup);

  for (int i = 0; i < nbGroup; i++) {
    threads.emplace_back([i, this]() { this->performGenerationGroup(i); });
  }

  for (std::thread &t : threads) {
    t.join();
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
