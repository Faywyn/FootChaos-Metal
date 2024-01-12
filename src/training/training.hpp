#include "../network/networksManager.hpp"

#include <filesystem>

namespace fs = std::filesystem;

class Training {
private:
  int nbNetwork;
  int nbLayer;
  int nbWeightPerNetwork;
  int *nbNeuronPerLayer;

  int groupSize;
  int **groups;
  int *nbNetworkPerGroup;

  NetworksManager *networksManager;

  float *result;

public:
  Training(int nbNetwork, int groupSize, int nbLayer, int *nbNeuronPerLayer);
  Training(fs::path path);
  ~Training();

  void save(fs::path path);

  void startTraining(int saveEveryX, int nbGeneration);
  void threadTraining();
  void createGroups();
  void performTrain();
};
