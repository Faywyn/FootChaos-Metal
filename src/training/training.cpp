#include "training.hpp"
#include "../utils.hpp"
#include "config.hpp"

#include <cstdlib>
#include <iostream>

/// Create new training
/// Parameters:
///  - nbNetwork
///  - groupSize
///  - nbLayer
///  - *nbNeuronPerLayer
///  - path
Training::Training(int nbNetwork, int groupSize, int nbLayer,
                   int *nbNeuronPerLayer, fs::path path) {

  if (nbNetwork % groupSize != 0)
    throw std::invalid_argument("nbNetwork % groupSize != 0");

  networksManager =
      new NetworksManager(nbNetwork, groupSize, nbLayer, nbNeuronPerLayer);
  this->path = path;
  this->init();
}

/// Create training from data
Training::Training(fs::path path) {
  networksManager = new NetworksManager(path);
  this->path = path;
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

  delete networksManager;
}

/// Save data
void Training::save() { networksManager->saveNetworks(path); }

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

  // Display border line
  std::string line = std::string(120, '-');
  std::cout << "\033[" << STAT_TAB_START + 0 << ";0H" << line;
  std::cout << "\033[" << STAT_TAB_START + 2 << ";0H" << line;
  std::cout << "\033[" << STAT_TAB_START + NB_STATS + 5 << ";0H" << line;
  std::cout << "\033[" << STAT_TAB_START + NB_STATS + 7 << ";0H" << line;
  std::cout << std::endl;

  while (*nbGen_ < nbGeneration || nbGeneration == -1) {
    uint64_t timeStart = time();
    float avrg;
    float avrgPoints;
    float best;

    performTrain();
    float **score = getScore();
    getScoreData(score, &avrg, &best, &avrgPoints);

    // Get stats
    uint64_t duration = time() - timeStart;
    minTime = minTime > duration ? duration : minTime;
    maxTime = maxTime < duration ? duration : maxTime;

    // Display stats
    printOldStat(STAT_TAB_START + 4 + (*nbGen_) % nbStat, *nbGen_,
                 time() - timeStart, avrg, avrgPoints, best,
                 networksManager->nbGame);

    std::cout << "\033[" << STAT_TAB_START + 6 + nbStat << ":0H\033[0m"
              << "  "
              << "Global Duration: "
              << round((float)(time() - globalTimeStart) / 1000, 100) << "s "
              << "Min: " << GREEN << round((float)minTime / 1000, 100) << "s "
              << RESET << "Max: " << RED << round((float)maxTime / 1000, 100)
              << "s " << RESET << std::endl;

    *nbGen_ += 1;

    // Saving networks
    if (*nbGen_ % saveEveryXn == 0)
      save();

    // Saving game
    if (*nbGen_ % 10 == 0)
      saveGame(score[0][1], score[0][1]);

    // Mutate network (score free !)
    mutate(score);
  }
}

float **Training::getScore() { return networksManager->getScore(); }

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

void Training::mutate(float **score) {
  // Creating new Networks (weakest remove)
  for (int i = 0; i < networksManager->nbNetwork * NEW_BLOOD_COEF; i++) {
    float r = (randomFloat() + 1) / 2;
    float k = (float)i /
              ((float)networksManager->nbNetwork * networksManager->nbNetwork);

    networksManager->randomizeNetwork(
        score[networksManager->nbNetwork - i - 1][0]);
  }

  // Copy the best networks (weakest remove)
  for (int i = 0; i < networksManager->nbNetwork * COPY_COEF; i++) {
    float r = (randomFloat() + 1) / 2;
    float k = (float)i / ((float)networksManager->nbNetwork * COPY_COEF);

    networksManager->copyNetwork(i,
                                 score[networksManager->nbNetwork - i - 1][0]);
  }

  // Mutate every network (weakest have more change to be mutate)
  for (int i = 0; i < networksManager->nbNetwork; i++) {
    float p = (float)(i + 1) / (float)(networksManager->nbNetwork + 1);
    p *= 10;
    p /= INPUT_LENGTH * networksManager->nbLayer;
    networksManager->mutateNetwork(score[i][0], p);

    free(score[i]);
  }
  free(score);
}

void Training::saveGame(int player1, int player2) {
  fs::path p = path.replace_extension("").string() + "_" +
               std::to_string(networksManager->nbGeneration) + ".txt";
  networksManager->saveGame(player1, player2, p);
}
