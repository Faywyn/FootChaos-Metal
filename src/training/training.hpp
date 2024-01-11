#include "../network/networksManager.hpp"

#include <filesystem>

namespace fs = std::filesystem;

class Training {
private:
  int nbNetwork;
  int nbLayer;
  int nbWeightPerNetwork;
  int *nbNeuronPerLayer;

  float *result;

  bool created = false;

public:
  Training();
  ~Training();

  void load(fs::path path);
  void save(fs::path path);
  void create(int nbNetwork, int nbLayer, int *nbNeuronPerLayer);

  void threadTraining(int *networks, int sizeOfGroups);
  int **createGroups(int sizeOfGroups);
  void performTrain(int sizeOfGroups);
  void startTraining(int saveEveryX, int sizeOfGroups);
};
