#include "training.hpp"
#include "../utils.hpp"
#include "config.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

/// Create new training
/// Parameters:
///  - nbNetwork
///  - groupSize
///  - nbLayer
///  - *nbNeuronPerLayer
///  - id (for saves)
Training::Training(int nbNetwork, int groupSize, int nbLayer,
                   int *nbNeuronPerLayer, int id) {

  if (nbNetwork % groupSize != 0)
    throw std::invalid_argument("nbNetwork % groupSize != 0");

  networksManager =
      new NetworksManager(nbNetwork, groupSize, nbLayer, nbNeuronPerLayer);
  this->path = TRAININGS_PATH / std::to_string(id);
  this->init();
}

/// Create training from data
/// Parameters:
///  - id (for saves)
Training::Training(int id) {
  this->path = TRAININGS_PATH / std::to_string(id);
  networksManager = new NetworksManager(path / "data.bin");
  this->init();
}

/// Init groups
void Training::init() {
  int nbNetwork = networksManager->nbNetwork;
  int groupSize = networksManager->groupSize;
  int nbGroup = nbNetwork / groupSize;

  groups = (int **)malloc(sizeof(int *) * nbGroup);
  nbNetworkPerGroup = (int *)malloc(sizeof(int) * nbGroup);
  for (int i = 0; i < nbGroup; i++) {
    groups[i] = (int *)malloc(sizeof(int) * groupSize);
  }

  // Create training directory
  if (fs::is_directory(TRAININGS_PATH) == false)
    fs::create_directory(TRAININGS_PATH);
  if (fs::is_directory(path) == false)
    fs::create_directory(path);
}

/// Destructor
Training::~Training() {
  int nbNetwork = networksManager->nbNetwork;
  int groupSize = networksManager->groupSize;
  int nbGroup = nbNetwork / groupSize;

  for (int i = 0; i < nbGroup; i++) {
    free(groups[i]);
  }
  free(groups);
  free(nbNetworkPerGroup);

  delete networksManager;
}

/// Save data
void Training::save() { networksManager->saveNetworks(path / "data.bin"); }

/// Create groups for the matchs
void Training::createGroups() {
  int nbNetwork = networksManager->nbNetwork;
  int groupSize = networksManager->groupSize;
  int nbGroup = nbNetwork / groupSize;
  for (int i = 0; i < nbGroup; i++) {
    nbNetworkPerGroup[i] = 0;
  }

  for (int i = 0; i < nbNetwork; i++) {
    int r;
    do {
      r = randomInt(0, nbGroup - 1);
    } while (nbNetworkPerGroup[r] == groupSize);

    groups[r][nbNetworkPerGroup[r]] = i;
    nbNetworkPerGroup[r] += 1;
  }
}

/// Perform every match with the differents groups
void Training::performTrain() {
  this->createGroups();
  networksManager->performGeneration(groups);
}

/// Start a train
/// Parameters:
///  - saveEveryXn: Save every X generations
///  - nbGeneration: -1 for timeless training
void Training::startTraining(int saveEveryXn, int nbGeneration) {
  int *nbGen_ = &networksManager->nbGeneration;

  system("clear");

  // Keeping all stats
  int nbStat = NB_STATS;
  uint64_t globalTimeStart = time();
  uint64_t minTime = globalTimeStart;
  uint64_t maxTime = 0;

  // Display some global stats
  std::cout << "\033[2;0H"
            << "  Networs: " << networksManager->nbNetwork << std::endl
            << "  Groups Size: " << networksManager->groupSize;
  std::cout << "\033[2;30H"
            << "Nb Match Per Gen: " << networksManager->nbGame;
  std::cout << "\033[3;30H"
            << "Nb Group: "
            << networksManager->nbNetwork / networksManager->groupSize;
  std::cout << "\033[2;60H"
            << "Weights: " << networksManager->nbWeightPerNetwork;

  // Display border line
  std::string line = std::string(120, '-');
  std::cout << "\033[" << STAT_TAB_START + 0 << ";0H" << line;
  std::cout << "\033[" << STAT_TAB_START + 2 << ";0H" << line;
  std::cout << "\033[" << STAT_TAB_START + NB_STATS + 5 << ";0H" << line;
  std::cout << "\033[" << STAT_TAB_START + NB_STATS + 7 << ";0H" << line;
  std::cout << std::endl;

  // Perform the gen
  while (*nbGen_ < nbGeneration || nbGeneration == -1) {
    // Some data for stats
    uint64_t timeStart = time();
    float avrg;
    float avrgPoints;
    float best;

    // ----- Doing the game -----
    performTrain();
    float **score = getScore();
    getScoreData(score, &avrg, &best, &avrgPoints);
    // ----- Doing the game -----

    // Get stats
    uint64_t duration = time() - timeStart;
    minTime = minTime > duration ? duration : minTime;
    maxTime = maxTime < duration ? duration : maxTime;
    saveMetrics(best, avrg, duration);

    // Display stats
    printOldStat(STAT_TAB_START + 4 + (*nbGen_) % nbStat, *nbGen_,
                 time() - timeStart, avrg, avrgPoints, best,
                 networksManager->nbGame);

    std::cout << "\033["
              << STAT_TAB_START + 4 + (*nbGen_ + NB_STATS - 1) % nbStat
              << ";1H " << std::endl;
    std::cout << "\033[" << STAT_TAB_START + 4 + (*nbGen_) % nbStat << ";1H>"
              << std::endl;

    // Display global stats
    printGlobalStat(time() - globalTimeStart, minTime, maxTime);

    *nbGen_ += 1;

    // Saving networks
    if (*nbGen_ % saveEveryXn == 0)
      save();

    // Saving game
    if (*nbGen_ % 10 == 0)
      saveGame(score[0][0], score[1][0]);

    // Mutate network (score is free !)
    mutate(score);
  }
}

/// Get score after gen
float **Training::getScore() { return networksManager->getScore(); }

/// Get score data (avrg, best ...)
/// Parameters:
///  - **score
///  - *avrg (average score, edit by function)
///  - *best (best score, edit by function)
///  - *avrgPoints (average point score, edit by function)
void Training::getScoreData(float **score, float *avrg, float *best,
                            float *avrgPoints) {
  int nbMatchPerNetwork = (networksManager->groupSize - 1);
  *avrg = 0;
  *avrgPoints = 0;
  *best = 0;

  for (int i = 0; i < networksManager->nbNetwork; i++) {
    float s = score[i][1] / (float)nbMatchPerNetwork;
    *avrg += s;
    *avrgPoints += abs(score[i][2]) / (2 * (float)nbMatchPerNetwork);
    *best = (*best < s) ? s : *best;
  }
  *avrg /= networksManager->nbNetwork;
  *avrgPoints /= networksManager->nbNetwork;
}

/// Mutate networks based on results
/// Parameters:
///  **score
void Training::mutate(float **score) {
  double nbMatchPerNetwork = (networksManager->groupSize - 1);
  double best = score[0][1] / nbMatchPerNetwork;
  int nbNetwork = networksManager->nbNetwork;

  // Creating new Networks (weakest remove)
  int nbNetworkNew = networksManager->nbNetwork * NEW_BLOOD_COEF;
  if (best < 0.5) {
    nbNetworkNew *= 0.5 / best;
    if (nbNetworkNew > nbNetwork / 4) {
      nbNetworkNew = nbNetwork / 4;
    }
  }
  for (int i = 0; i < nbNetworkNew; i++) {
    float r = abs(randomFloat());
    float k = (float)i / (float)nbNetworkNew;

    if (r > k)
      continue;

    int id = networksManager->nbNetwork - i - 1;
    networksManager->randomizeNetwork(score[id][0]);
  }

  // Copy the best networks (weakest remove)
  int nbNetworkCopy = networksManager->nbNetwork * COPY_COEF;
  for (int i = 0; i < nbNetworkCopy; i++) {
    float r = abs(randomFloat());
    float k = (float)i / (float)nbNetworkCopy;

    if (r > k)
      continue;

    int id = networksManager->nbNetwork - i - 1;
    networksManager->copyNetwork(i, score[id][0]);
  }

  std::vector<std::thread> threads;
  threads.reserve(NB_THREAD);

  // Mutate every network (weakest have more change to be mutate)
  for (int i = 0; i < NB_THREAD; i++) {
    threads.emplace_back([i, score, this]() {
      int n = networksManager->nbNetwork;
      int start = i * n / NB_THREAD;
      int end = (i == NB_THREAD - 1) ? n : (i + 1) * n / NB_THREAD;
      for (int i = start; i < end; i++) {
        float p = (float)(i + 1) / (float)(networksManager->nbNetwork);
        p = p * p;
        // "NB_WEIGHT_CHANGE" weights change per network on average (depending
        // on i)
        p /= (float)networksManager->nbWeightPerNetwork / NB_WEIGHT_CHANGE;
        networksManager->mutateNetwork(score[i][0], p);

        free(score[i]);
      }
    });
  }

  for (std::thread &t : threads) {
    t.join();
  }

  free(score);
}

/// Do a game and save it
/// Parameters:
///  - player1
///  - player2
void Training::saveGame(int player1, int player2) {
  fs::path p =
      path / ("M" + std::to_string(networksManager->nbGeneration) + ".csv");
  networksManager->saveGame(player1, player2, p);
}

/// Append stats to metrics file
/// Parameters:
///  - best
///  - avrg
///  - time (gen time)
void Training::saveMetrics(float best, float avrg, float time) {
  fs::path p = path / ("metrics.csv");

  std::ofstream csv(p, std::ios::app);
  float nbMatchSec = 1000 * networksManager->nbGame / time;
  csv << networksManager->nbGeneration << ";" << best << ";" << avrg << ";"
      << time << ";" << networksManager->nbNetwork << ";"
      << networksManager->groupSize << ";" << nbMatchSec << "\n";
  csv.close();
}
