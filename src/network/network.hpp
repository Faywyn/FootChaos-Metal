#include "../config.hpp"

class Network {
private:
  float **weight = nullptr; // weight[i] all the weight of layer i
  int *sizeOfLayer = nullptr;
  int *sizeOfLayerWeight = nullptr;
  int nbLayer = 0;

public:
  Network();
  ~Network();

  void createRandom(int *sizeOfLayer, int nbLayer);
  void load(float **weight, int *sizeOfLayer, int nbLayer);
  void display();
};
