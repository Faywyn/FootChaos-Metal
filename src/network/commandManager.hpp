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
  MTL::ComputeCommandEncoder *commandEncoderT;
  MTL::ComputeCommandEncoder *commandEncoderN;

  int nbLayer;

public:
  CommandManager(int _nbGame, int tickId, NetworksManager *manager);
  ~CommandManager();

  MTL::Size getThreadGroupSize(int nbExec, MTL::ComputePipelineState *function);
  void compute();
};
