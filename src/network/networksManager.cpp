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
#include "commandManager.hpp"
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
    file.read((char *)&(nbNeuronPerLayer[i]), sizeof(int));
  }

  this->initSystem();
  this->initBuffer();
  this->initGames();

  for (int i = 1; i < nbLayer; i++) {
    int nbWeigth = nbNetwork * nbNeuronPerLayer[i] * nbNeuronPerLayer[i - 1];
    float *cont = (float *)weights[i]->contents();
    for (int j = 0; j < nbWeigth; j++) {
      file.read((char *)&cont[j], sizeof(float));
    }
  }
  file.close();
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
  this->initGames();

  // Generate random weights
  for (int i = 1; i < nbLayer; i++) {
    int nbWeigth = nbNetwork * nbNeuronPerLayer[i] * nbNeuronPerLayer[i - 1];
    float *values = (float *)weights[i]->contents();
    for (int j = 0; j < nbWeigth; j++) {
      values[j] = randomFloat();
    }
  }
}

/// NetworksManager destructor
NetworksManager::~NetworksManager() {
  if (weightFunctionPSO != nullptr)
    weightFunctionPSO->release();
  if (network1 != nullptr)
    network1->release();
  if (network2 != nullptr)
    network2->release();
  if (commandQueue != nullptr)
    commandQueue->release();
  if (device != nullptr)
    device->release();

  free(nbNeuronPerLayer);
  for (int i = 1; i < nbLayer; i++) {
    weights[i]->release();
    data[i]->release();
    result[0][i]->release();
    result[1][i]->release();
  }
  result[0][0]->release();
  result[1][0]->release();
  free(result[0]);
  free(result[1]);
  free(result);
  free(data);
  free(weights);
  for (int i = 0; i < nbGame; i++) {
    delete games[i];
  }
  free(games);
}

/// Create buffers for games and networks
void NetworksManager::initBuffer() {
  this->nbGame = (nbNetwork / groupSize) * (groupSize - 1) * (groupSize) / 2;
  this->network1 =
      device->newBuffer(sizeof(int) * nbGame, MTL::ResourceStorageModeShared);
  this->network2 =
      device->newBuffer(sizeof(int) * nbGame, MTL::ResourceStorageModeShared);
  this->data = (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * nbLayer);
  this->result = (MTL::Buffer ***)malloc(sizeof(MTL::Buffer **) * 2);
  this->result[0] = (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * nbLayer);
  this->result[1] = (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * nbLayer);

  nbWeightPerNetwork = 0;
  weights = (MTL::Buffer **)malloc(sizeof(MTL::Buffer *) * nbLayer);
  for (int i = 1; i < nbLayer; i++) {
    // Set weight array
    int nbWeigth = nbNeuronPerLayer[i] * nbNeuronPerLayer[i - 1];
    nbWeightPerNetwork += nbWeigth;
    weights[i] = device->newBuffer(sizeof(float) * nbNetwork * nbWeigth,
                                   MTL::ResourceStorageModeShared);
    // Set data Buffers
    data[i] =
        device->newBuffer(sizeof(int) * 2, MTL::ResourceStorageModeShared);
    int *cont = (int *)data[i]->contents();
    cont[0] = nbNeuronPerLayer[i];
    cont[1] = nbNeuronPerLayer[i - 1];
  }

  for (int k = 0; k < 2; k++) {
    for (int i = 1; i < nbLayer - 1; i++) {
      // Set result Buffers
      result[k][i] =
          device->newBuffer(sizeof(float) * nbNeuronPerLayer[i] * nbGame * 2,
                            MTL::ResourceStorageModePrivate);
    }
    result[k][0] =
        device->newBuffer(sizeof(float) * nbNeuronPerLayer[0] * nbGame * 2,
                          MTL::ResourceStorageModeShared);
    result[k][nbLayer - 1] = device->newBuffer(
        sizeof(float) * nbNeuronPerLayer[nbLayer - 1] * nbGame * 2,
        MTL::ResourceStorageModeShared);
  }
}

/// Init device, metals functions, ...
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

  // Create command queue
  commandQueue = device->newCommandQueue();
  if (commandQueue == nullptr)
    throw std::invalid_argument("Impossible to create command queue");
}

/// Create all the games
void NetworksManager::initGames() {
  games = (FootChaos **)malloc(sizeof(FootChaos *) * nbGame);
  for (int i = 0; i < nbGame; i++) {
    games[i] = new FootChaos(i, true, "");
  }
}

/// Save networks and generation data to file
/// Parameters:
///  - path
void NetworksManager::saveNetworks(fs::path path) {
  std::ofstream file;
  file.open(path, std::ios::binary);

  // Save some data
  file.write((char *)&nbGeneration, sizeof(int));
  file.write((char *)&nbNetwork, sizeof(int));
  file.write((char *)&nbLayer, sizeof(int));
  file.write((char *)&groupSize, sizeof(int));

  // Save nb network per layer
  for (int i = 0; i < nbLayer; i++) {
    file.write((char *)&nbNeuronPerLayer[i], sizeof(int));
  }

  // Save every weight
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
///  - _nbGame (in case we want 1 for saving)
void NetworksManager::computeNetworks(int _nbGame, int tickId) {
  CommandManager commandManager = CommandManager(_nbGame, tickId, this);
  commandManager.compute();
}

/// Create matchs
void NetworksManager::initGeneration() {
  // Witch network in witch match
  int *contNet1 = (int *)network1->contents();
  int *contNet2 = (int *)network2->contents();

  int _nbGame = 0;
  for (int iG = 0; iG < nbNetwork / groupSize; iG++) {
    for (int iP = 0; iP < groupSize; iP++) {
      for (int jP = iP + 1; jP < groupSize; jP++) {
        // Reset games
        games[_nbGame]->resetPosition();
        games[_nbGame]->scoreTeam1 = 0;
        games[_nbGame]->scoreTeam2 = 0;
        games[_nbGame]->scoreTeam1Pos = 0;
        games[_nbGame]->scoreTeam2Pos = 0;

        contNet1[_nbGame] = groups[iG][iP];
        contNet2[_nbGame] = groups[iG][jP];

        _nbGame++;
      }
    }
  }

  // Just in case
  assert(_nbGame == nbGame);
}

void NetworksManager::setInputs(int tickId, std::vector<std::thread> *threads) {
  float *inputsCont = (float *)result[tickId % 2][0]->contents();

  // Set inputs
  for (int i = 0; i < NB_THREAD; i++) {
    threads->emplace_back([i, inputsCont, this]() {
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
}

/// Tick with the result
/// Parameters:
///  - nbGame
///  - tickId
void NetworksManager::performTickGeneration(int tickId,
                                            std::vector<std::thread> *threads) {
  // Dispatch with threads
  computeNetworks(nbGame, tickId);
  MTL::Buffer *res = result[tickId % 2][nbLayer - 1];
  float *resCont = (float *)res->contents();
  for (int i = 0; i < NB_THREAD; i++) {
    threads->emplace_back([i, resCont, this]() {
      int start = i * nbGame / NB_THREAD;
      int end = (i == NB_THREAD - 1) ? nbGame : (i + 1) * nbGame / NB_THREAD;
      for (int j = start; j < end; j++) {

        games[j]->tick(resCont);
      }
    });
  }
}

/// Perform every tick needed for a full game
/// Parameters:
///  - **groups
void NetworksManager::performGeneration(int **groups) {
  // Init matchs and groups
  this->groups = groups;
  this->initGeneration();

  std::vector<std::thread> threadsInputs;
  setInputs(0, &threadsInputs);
  for (std::thread &t : threadsInputs) {
    t.join();
  }

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
    std::vector<std::thread> threadsTick;
    std::vector<std::thread> threadsInputs;

    performTickGeneration(i, &threadsTick);
    for (std::thread &t : threadsTick) {
      t.join();
    }

    setInputs(i + 1, &threadsInputs);
    for (std::thread &t : threadsInputs) {
      t.join();
    }
  }
}

/// Get the result of gen
float **NetworksManager::getScore() {
  float **res = (float **)malloc(sizeof(float *) * nbNetwork);

  // res[i][0]: network id
  // res[i][1]: score
  for (int i = 0; i < nbNetwork; i++) {
    res[i] = (float *)malloc(sizeof(float) * 3);
    res[i][0] = i;
    res[i][1] = 0.0f;
    res[i][2] = 0.0f;
  }

  // Sum each score
  int _nbGame = 0;
  for (int iG = 0; iG < nbNetwork / groupSize; iG++) {
    for (int iP = 0; iP < groupSize; iP++) {
      for (int jP = iP + 1; jP < groupSize; jP++) {
        res[groups[iG][iP]][1] += games[_nbGame]->scoreTeam1;
        res[groups[iG][jP]][1] += games[_nbGame]->scoreTeam2;

        res[groups[iG][iP]][2] += games[_nbGame]->scoreTeam1Pos;
        res[groups[iG][jP]][2] += games[_nbGame]->scoreTeam2Pos;

        _nbGame++;
      }
    }
  }
  // Sort the results (randomize when equal)
  std::sort(res, res + nbNetwork, compareRdm);
  std::sort(res, res + nbNetwork, compare);

  return res;
}

/// Copy network's weights to another
/// Parameters:
///  - from
///  - to
void NetworksManager::copyNetwork(int from, int to) {
  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
    int nbWeigth = nbNeuronPerLayer[iLayer] * nbNeuronPerLayer[iLayer - 1];
    int start = from * nbWeigth;
    int end = (from + 1) * nbWeigth;
    int diff = (to - from) * nbWeigth;

    float *cont = (float *)weights[iLayer]->contents();
    for (int i = start; i < end; i++) {
      cont[i + diff] = cont[i];
    }
  }
}

/// Randomize a network
/// Parameters:
///  - id
void NetworksManager::randomizeNetwork(int id) {
  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
    int nbWeigth = nbNeuronPerLayer[iLayer] * nbNeuronPerLayer[iLayer - 1];
    int start = id * nbWeigth;
    int end = (id + 1) * nbWeigth;

    float *cont = (float *)weights[iLayer]->contents();
    for (int i = start; i < end; i++) {
      cont[i] = randomFloat();
    }
  }
}

/// Mutate network
/// Parameters:
///  - id
///  - p: proba to change a weight
void NetworksManager::mutateNetwork(int id, float p) {
  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
    int nbWeigth = nbNeuronPerLayer[iLayer] * nbNeuronPerLayer[iLayer - 1];
    int start = id * nbWeigth;
    int end = (id + 1) * nbWeigth;

    float *cont = (float *)weights[iLayer]->contents();
    for (int i = start; i < end; i++) {
      float r = abs(randomFloat());
      if (r > p)
        continue;
      cont[i] += randomFloat();
    }
  }
}

/// Perform a game and save it, doing the same but with 1 group of size 2
/// Parameters:
///  - player1
///  - player2
///  - path (where to save the game)
void NetworksManager::saveGame(int player1, int player2, fs::path path) {
  // Add "path" in order to save the game
  FootChaos game = FootChaos(0, true, path);

  float startIndex[2];
  startIndex[0] = 0;
  startIndex[1] = INPUT_LENGTH;

  // Set up the group (no need for realock)
  int *net1Cont = (int *)network1->contents();
  int *net2Cont = (int *)network2->contents();

  net1Cont[0] = player1;
  net2Cont[0] = player2;

  MTL::Buffer *res = result[0][nbLayer - 1];
  MTL::Buffer *input = result[0][0];
  float *resCont = (float *)res->contents();
  float *inputsCont = (float *)input->contents();

  // Performing the game
  for (int iTick = 0; iTick < GAME_LENGTH * TICKS_SECOND; iTick++) {
    game.setInputs(inputsCont, startIndex);
    computeNetworks(1, 0);

    game.tick(resCont);
  }
}
