#include "config.hpp"
#include <cstring>
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "networksManager.hpp"
#include <iostream>

NetworksManager::NetworksManager(){};

NetworksManager::NetworksManager(Network **networks, int nbNetwork,
                                 Network *networkModel) {
  this->networks = networks;
  this->nbNetwork = nbNetwork;
  this->networkModel = networkModel;

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

  created = true;
}

NetworksManager::~NetworksManager() {
  if (created == false)
    return;
  if (weightFunctionPSO != nullptr)
    weightFunctionPSO->release();
  if (activationFunctionPSO != nullptr)
    activationFunctionPSO->release();
}

Network **NetworksManager::getNetworks() { return networks; }
Network *NetworksManager::getNetwork(int id) { return networks[id]; }
int NetworksManager::getNbNetwork() { return nbNetwork; }

float *NetworksManager::getWeights() { return networkModel->getWeights(); }
int *NetworksManager::getNbNeuronPerLayer() {
  return networkModel->getNbNeuronPerLayer();
}
int NetworksManager::getNbNeuron() { return networkModel->getNbNeuron(); }
int NetworksManager::getNbLayer() { return networkModel->getNbLayer(); }
int NetworksManager::getNbWeight() { return networkModel->getNbWeight(); }

void NetworksManager::initWeightsBuffers(
    MTL::ComputeCommandEncoder *computeEncoder, int layerIndex, float *inputs) {

  int *nbNeuronPerLayer = networkModel->getNbNeuronPerLayer();
  int sizePreviousLayer = nbNeuronPerLayer[layerIndex - 1];
  int sizeLayer = nbNeuronPerLayer[layerIndex];

  bufferInputs =
      device->newBuffer(sizeof(float) * nbGame * 2 * sizePreviousLayer,
                        MTL::ResourceStorageModeShared);
  bufferResult = device->newBuffer(sizeof(float) * nbGame * 2 * sizeLayer,
                                   MTL::ResourceStorageModeShared);
  bufferNetworksWeights = device->newBuffer(sizeof(float) * nbGame * 2 *
                                                sizePreviousLayer * sizeLayer,
                                            MTL::ResourceStorageModeShared);
  bufferSizeLayer =
      device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
  bufferSizePreviousLayer =
      device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);

  float *inputsContent = (float *)bufferInputs->contents();
  float *resultsContent = (float *)bufferResult->contents();
  float *networksWeightsContent = (float *)bufferNetworksWeights->contents();
  int *sizeLayerContent = (int *)bufferSizeLayer->contents();
  int *sizePreviousLayerContent = (int *)bufferSizePreviousLayer->contents();

  assert(inputsContent);
  assert(resultsContent);
  assert(networksWeightsContent);
  assert(sizeLayerContent);
  assert(sizePreviousLayerContent);

  for (int i = 0; i < nbGame * 2 * sizePreviousLayer; i++) {
    inputsContent[i] = inputs[i];
  }
  for (int i = 0; i < nbGame * 2 * sizeLayer; i++) {
    resultsContent[i] = 0;
  }
  std::cout << "FF" << std::endl;

  for (int i = 0; i < nbGame * 2; i++) {
    float *networkLayerWeights =
        networks[networksInGame[i]]->getWeightLayer(layerIndex);
    for (int j = 0; j < sizePreviousLayer * sizeLayer; j++) {
      networksWeightsContent[i * sizePreviousLayer * sizeLayer + j] =
          networkLayerWeights[j];
    }
    free(networkLayerWeights);
  }

  sizeLayerContent[0] = sizeLayer;
  sizePreviousLayerContent[0] = sizePreviousLayer;

  computeEncoder->setBuffer(bufferInputs, 0, 0);
  computeEncoder->setBuffer(bufferResult, 0, 1);
  computeEncoder->setBuffer(bufferNetworksWeights, 0, 2);
  computeEncoder->setBuffer(bufferSizeLayer, 0, 3);
  computeEncoder->setBuffer(bufferSizePreviousLayer, 0, 4);
}

void NetworksManager::freeBuffers() {
  bufferInputs->release();
  bufferResult->release();
  bufferResultActivation->release();
  bufferNetworksWeights->release();
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

float *NetworksManager::computeWeight(float *inputs) {
  MTL::CommandQueue *commandeQueueW = device->newCommandQueue();
  MTL::CommandQueue *commandeQueueA = device->newCommandQueue();

  if (commandeQueueW == nullptr || commandeQueueA == nullptr)
    throw std::invalid_argument("Failed to find the command queue");

  int nbLayer = getNbLayer();
  int *nbNeuronPerLayer = getNbNeuronPerLayer();
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

    int nbWeightsLayer = nbNetwork * sizeLayer * sizePreviousLayer;
    MTL::Size gridSize = MTL::Size::Make(nbWeightsLayer, 1, 1);

    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize =
        weightFunctionPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > nbWeightsLayer) {
      threadGroupSize = nbWeightsLayer;
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

    float *contentRes = (float *)bufferResult->contents();
    bufferResultActivation = device->newBuffer(
        sizeof(float) * sizeLayer * nbNetwork, MTL::ResourceStorageModeShared);
    float *contentResActivation = (float *)bufferResultActivation->contents();
    for (int i = 0; i < sizeLayer * nbNetwork; i++) {
      contentResActivation[i] = contentRes[i];
    }
    computeEncoderA->setBuffer(bufferResultActivation, 0, 0);
    computeActivation(computeEncoderA, commandBufferA, sizeLayer);

    commandBufferA->release();
    computeEncoderA->release();

    free(inputs);
    inputs = (float *)malloc(sizeof(float) * sizeLayer * nbNetwork);

    std::memcpy(inputs, contentResActivation,
                sizeof(float) * sizeLayer * nbNetwork);

    // for (int i = 0; i < sizeLayer * nbNetwork; i++) {
    //   inputs[i] = contentResActivation[i];
    // }
    freeBuffers();
  }

  commandeQueueA->release();
  commandeQueueW->release();

  return inputs;
}

void NetworksManager::initGeneration() {
  nbGame = nbNetwork * (nbNetwork - 1) / 2;
  games = (FootChaos **)malloc(sizeof(FootChaos *) * nbGame);
  networksInGame = (int *)malloc(sizeof(int) * nbGame * 2);

  assert(games);
  assert(networksInGame);

  int _nbGame = 0;
  for (int i = 0; i < nbNetwork; i++) {
    for (int j = i + 1; j < nbNetwork; j++) {
      games[_nbGame] = new FootChaos(2, _nbGame);

      networksInGame[_nbGame * 2 + 0] = i;
      networksInGame[_nbGame * 2 + 1] = j;

      _nbGame += 1;
    }
  }
  std::cout << "AA" << std::endl;
}

void NetworksManager::performTickGeneration() {
  float *networkInput =
      (float *)malloc(sizeof(float) * nbGame * INPUT_LENGTH * 2);

  for (int i = 0; i < nbGame; i++) {
    FootChaos *game = games[i];

    float *startIndex = (float *)malloc(sizeof(float) * 2);
    startIndex[0] = (i * 2 + 0) * INPUT_LENGTH;
    startIndex[1] = (i * 2 + 1) * INPUT_LENGTH;

    game->setInputs(networkInput, startIndex);
    free(startIndex);
  }

  std::cout << "BB" << std::endl;

  float *res = computeWeight(networkInput);
  for (int i = 0; i < nbGame; i++) {
    games[i]->tick(res);
  }
}

void NetworksManager::performGeneration() {
  std::cout << "T" << std::endl;
  this->initGeneration();

  for (int i = 0; i < GAME_LENGTH * TICKS_SECOND; i++) {
  // NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    std::cout << "-----" << std::endl;
    performTickGeneration();
    std::cout << "huidfazhidza" << std::endl;
    // std::cout << i << "/" << GAME_LENGTH * TICKS_SECOND << std::endl;
    // pool->release();
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
