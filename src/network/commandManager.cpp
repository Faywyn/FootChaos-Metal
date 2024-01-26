#include "commandManager.hpp"

CommandManager::CommandManager(int _nbGame, int tickId,
                               NetworksManager *manager) {
  MTL::CommandQueue *commandQueue = manager->commandQueue;
  MTL::ComputePipelineState *weightFunctionPSO = manager->weightFunctionPSO;
  MTL::Buffer **weights = manager->weights;
  MTL::Buffer **data = manager->data;
  MTL::Buffer ***result = manager->result;
  MTL::Buffer *network1 = manager->network1;
  MTL::Buffer *network2 = manager->network2;
  int *nbNeuronPerLayer = manager->nbNeuronPerLayer;
  nbLayer = manager->nbLayer;

  // Create command buffer
  commandBuffer = commandQueue->commandBuffer();
  if (commandBuffer == nullptr)
    throw std::invalid_argument("Impossible to create command buffer");

  // List all buffer and encoder used for each layer (free later)
  encoders = (MTL::CommandEncoder **)malloc(sizeof(MTL::CommandEncoder *) *
                                            (nbLayer - 1));

  // For each layer add instructions
  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
    int sizePreviousLayer = nbNeuronPerLayer[iLayer - 1];
    int sizeLayer = nbNeuronPerLayer[iLayer];

    // Nb thread
    int nbExec = _nbGame * sizeLayer * 2;
    MTL::Size gridSize = MTL::Size::Make(nbExec, 1, 1);

    NS::UInteger threadGroupSize =
        weightFunctionPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > nbExec) {
      threadGroupSize = nbExec;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Command encoder
    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();
    if (computeEncoder == nullptr)
      throw std::invalid_argument("Impossible to create compute encoder (" +
                                  std::to_string(iLayer) + ")");

    // Set function
    computeEncoder->setComputePipelineState(weightFunctionPSO);

    // Set function params'
    computeEncoder->setBuffer(result[tickId % 2][iLayer - 1], 0, 0);
    computeEncoder->setBuffer(result[tickId % 2][iLayer], 0, 1);
    computeEncoder->setBuffer(weights[iLayer], 0, 2);
    computeEncoder->setBuffer(network1, 0, 3);
    computeEncoder->setBuffer(network2, 0, 4);
    computeEncoder->setBuffer(data[iLayer], 0, 5);

    // Encode the command
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    // Release at the end
    encoders[iLayer - 1] = computeEncoder;
  }
}

CommandManager::~CommandManager() {
  // Release buffers and encoders
  for (int i = 0; i < nbLayer - 1; i++) {
    encoders[i]->release();
  }
  free(encoders);
  commandBuffer->release();
}

void CommandManager::compute() {
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}
