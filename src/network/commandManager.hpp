#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "networksManager.hpp"

class NetworksManager;
class CommandManager {
private:
  MTL::CommandBuffer *commandBuffer;
  MTL::CommandEncoder **encoders;

  int nbLayer;

public:
  CommandManager(int _nbGame, int tickId, NetworksManager *manager);
  ~CommandManager();

  void compute();
};
