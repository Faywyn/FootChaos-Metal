#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
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

void NetworksManager::freeBuffers() {
  return;
  bufferResult->release();
  bufferSizeLayer->release();
  bufferSizePreviousLayer->release();
}

void NetworksManager::computeActivation(
    MTL::ComputeCommandEncoder *computeEncoderA,
    MTL::CommandBuffer *commandBufferA, int length) {

  MTL::Size gridSize = MTL::Size::Make(length * nbNetwork, 1, 1);

  // Calculate a threadgroup size.
  NS::UInteger threadGroupSize =
      weightFunctionPSO->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > length * nbNetwork) {
    threadGroupSize = length * nbNetwork;
  }
  MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

  computeEncoderA->dispatchThreads(
      gridSize, threadgroupSize); // Encode the compute command.
  computeEncoderA->endEncoding(); // End the compute pass.
  commandBufferA->commit();       // Execute the command.
  commandBufferA->waitUntilCompleted();
}

float *NetworksManager::computeWeight(MTL::Buffer *inputs) {
  MTL::CommandQueue *commandeQueueW = device->newCommandQueue();
  MTL::CommandQueue *commandeQueueA = device->newCommandQueue();

  if (commandeQueueW == nullptr || commandeQueueA == nullptr)
    throw std::invalid_argument("Failed to find the command queue");

  bufferSizeLayer =
      device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
  bufferSizePreviousLayer =
      device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);

  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
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

    // -----
    // Compute the layer values
    // -----

    initWeightsBuffers(computeEncoderW, iLayer, inputs);

    int nbExec = nbGame * sizeLayer * sizePreviousLayer * 2;
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
    for (int iP = 0; iP < groupSize; iP++) {
      for (int jP = iP + 1; jP < groupSize; jP++) {
        games[_nbGame] = new FootChaos(2, _nbGame);

        contNet1[_nbGame] = groups[iG][iP];
        contNet2[_nbGame] = groups[iG][jP];

        _nbGame++;
      }
    }
  }

  std::cout << _nbGame << "-" << nbGame << std::endl;

  assert(_nbGame == nbGame);
}

void NetworksManager::performTickGeneration() {
  MTL::Buffer *inputs =
      device->newBuffer(sizeof(float) * INPUT_LENGTH * nbGame * 2,
                        MTL::ResourceStorageModeShared);

  float *inputsCont = (float *)inputs->contents();

  for (int i = 0; i < nbGame; i++) {
    float *startIndex = (float *)malloc(sizeof(float) * 2);
    startIndex[0] = (i * 2 + 0) * INPUT_LENGTH;
    startIndex[1] = (i * 2 + 1) * INPUT_LENGTH;

    games[i]->setInputs(inputsCont, startIndex);
    free(startIndex);
  }

  float *res = computeWeight(inputs);
  for (int i = 0; i < nbGame; i++) {
    games[i]->tick(res);
  }
}

void NetworksManager::performGeneration(int **groups) {
  this->groups = groups;
  this->initGeneration();

  std::cout << "Nb Group: " << nbNetwork / groupSize
            << "\nNb Network: " << nbNetwork << "\nNb Game: " << nbGame
            << std::endl;

  for (int i = 0; i < GAME_LENGTH * TICKS_SECOND; i++) {
    performTickGeneration();
    if (i % ((GAME_LENGTH * TICKS_SECOND + 1) / 10) == 0)
      std::cout << i << "/" << GAME_LENGTH * TICKS_SECOND << std::endl;
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
