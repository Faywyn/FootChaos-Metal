#include "../network/networksManager.hpp"

#include <filesystem>

namespace fs = std::filesystem;

class Training {
private:
  fs::path path;

  NetworksManager *networksManager;

  int **groups;
  int *nbNetworkPerGroup;

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
  void mutate(float **score);

  void createGroups();
  void performTrain();
};
