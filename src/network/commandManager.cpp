#include "commandManager.hpp"
#include "config.hpp"
#include <iostream>

CommandManager::CommandManager(int _nbGame, int tickId,
                               NetworksManager *manager) {
  MTL::CommandQueue *commandQueue = manager->commandQueue;
  MTL::ComputePipelineState *weightFunctionPSO = manager->weightFunctionPSO;
  MTL::ComputePipelineState *trigoFunctionPSO = manager->dataTrigFunctionPSO;
  MTL::ComputePipelineState *normFunctionPSO = manager->dataNormFunctionPSO;
  MTL::Buffer **weights = manager->weights;
  MTL::Buffer **data = manager->data;
  MTL::Buffer *dataInput = manager->inputData;
  MTL::Buffer *inputDataNorm = manager->inputDataNorm;
  MTL::Buffer *inputDataTrig = manager->inputDataTrig;
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

  // Add normalize and cos/sin function
  int nbExecN = _nbGame * INPUT_NORM_DATA_LENGTH * 2;
  int nbExecT = _nbGame * INPUT_TRIG_DATA_LENGTH * 2;

  MTL::Size gridSizeN = MTL::Size::Make(nbExecN, 1, 1);
  MTL::Size gridSizeT = MTL::Size::Make(nbExecT, 1, 1);

  MTL::Size threadgroupSizeN = getThreadGroupSize(nbExecN, normFunctionPSO);
  MTL::Size threadgroupSizeT = getThreadGroupSize(nbExecT, trigoFunctionPSO);

  // Command encoder
  commandEncoderN = commandBuffer->computeCommandEncoder();
  if (commandEncoderN == nullptr)
    throw std::invalid_argument("Impossible to create command encoder");

  commandEncoderN->setComputePipelineState(normFunctionPSO);

  // Set function params'
  commandEncoderN->setBuffer(inputDataNorm, 0, 0);
  commandEncoderN->setBuffer(result[tickId % 2][0], 0, 1);
  commandEncoderN->setBuffer(dataInput, 0, 2);

  // Encode the command
  commandEncoderN->dispatchThreads(gridSizeN, threadgroupSizeN);
  commandEncoderN->endEncoding();

  // Command encoder
  commandEncoderT = commandBuffer->computeCommandEncoder();
  if (commandEncoderT == nullptr)
    throw std::invalid_argument("Impossible to create command encoder");

  commandEncoderT->setComputePipelineState(trigoFunctionPSO);

  // Set function params'
  commandEncoderT->setBuffer(inputDataTrig, 0, 0);
  commandEncoderT->setBuffer(result[tickId % 2][0], 0, 1);
  commandEncoderT->setBuffer(dataInput, 0, 2);

  // Encode the command
  commandEncoderT->dispatchThreads(gridSizeT, threadgroupSizeT);
  commandEncoderT->endEncoding();

  // For each layer add instructions
  for (int iLayer = 1; iLayer < nbLayer; iLayer++) {
    int sizePreviousLayer = nbNeuronPerLayer[iLayer - 1];
    int sizeLayer = nbNeuronPerLayer[iLayer];

    // Nb thread
    int nbExec = _nbGame * sizeLayer * 2;
    MTL::Size gridSize = MTL::Size::Make(nbExec, 1, 1);
    MTL::Size threadgroupSize = getThreadGroupSize(nbExec, weightFunctionPSO);

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

MTL::Size
CommandManager::getThreadGroupSize(int nbExec,
                                   MTL::ComputePipelineState *function) {
  MTL::Size gridSize = MTL::Size::Make(nbExec, 1, 1);

  NS::UInteger threadGroupSize = function->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > nbExec) {
    threadGroupSize = nbExec;
  }
  return MTL::Size::Make(threadGroupSize, 1, 1);
}

CommandManager::~CommandManager() {
  // Release buffers and encoders
  for (int i = 0; i < nbLayer - 1; i++) {
    encoders[i]->release();
  }
  free(encoders);
  commandBuffer->release();
  commandEncoderN->release();
  commandEncoderT->release();
}

void CommandManager::compute() {
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}
