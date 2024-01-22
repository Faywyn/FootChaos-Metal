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
           fs::path path);
  Training(fs::path path);
  ~Training();

  // Create groups
  void init();

  void save();

  void startTraining(int saveEveryX, int nbGeneration);
  float **getScore();
  void saveGame(int player1, int player2);
  void mutate(float **score);

  void createGroups();
  void performTrain();
};
