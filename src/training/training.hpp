#pragma once

#include "../network/networksManager.hpp"

#include <filesystem>
namespace fs = std::filesystem;

class Training {
private:
  fs::path path;

  NetworksManager *networksManager;

  int **groups;
  int *nbNetworkPerGroup;
  void getScoreData(float **score, float *avrg, float *best, float *avrgPoints);

public:
  Training(int nbNetwork, int groupSize, int nbLayer, int *nbNeuronPerLayer,
           int id);
  Training(int id);
  ~Training();

  void init();
  void save();
  void createGroups();

  void performTrain();
  void startTraining(int saveEveryX, int nbGeneration);
  float **getScore();
  void mutate(float **score);

  void saveMetrics(float best, float avrg, float time);
  void saveGame(int player1, int player2);
};
