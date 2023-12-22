#include "network.hpp"

class Manager {
private:
  // Networks
  Network **networks = nullptr;
  int nbNetwork = 0;

  // Metal

private:
  Manager();
  ~Manager();

  void create(int nbNetwork, int *sizeOfLayer, int nbLayer);
};
