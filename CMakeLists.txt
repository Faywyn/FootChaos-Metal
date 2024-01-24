cmake_minimum_required(VERSION 3.26)
project(FootChaosExe)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# set(CMAKE_BUILD_TYPE Debug)

include_directories("src")

find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)

set(SRC_FILES
  src/main.cpp
  src/config.hpp
  src/network/networksManager.hpp
  src/network/networksManager.cpp
  src/footchaos/footchaos.hpp
  src/footchaos/footchaos.cpp
  src/footchaos/ball.cpp
  src/footchaos/car.cpp
  src/training/training.hpp
  src/training/training.cpp
  src/utils.hpp
  src/utils.cpp
)

file(GLOB SRC_SHADERS_FILES CONFIGURE_DEPENDS
        "src/*/*.metal"
        "src/*.metal"
        )

add_custom_command(
  OUTPUT footChaos.ir
  COMMAND xcrun -sdk macosx metal -o footChaos.ir -c src/network/networksCompute.metal
  DEPENDS ${SRC_SHADERS_FILES}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_command(
  OUTPUT shaders.metallib
  COMMAND xcrun -sdk macosx metallib -o shaders.metallib footChaos.ir
  DEPENDS footChaos.ir
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

include_directories(~/.vcpkg-clion/vcpkg/installed/arm64-osx/include)
link_directories(~/.vcpkg-clion/vcpkg/installed/arm64-osx/lib)

add_executable(FootChaosExe ${SRC_FILES} shaders.metallib)

target_include_directories(FootChaosExe
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp
)

target_link_libraries(FootChaosExe
  ${APPLE_FWK_FOUNDATION}
  ${APPLE_FWK_QUARTZ_CORE}
  ${APPLE_FWK_METAL}
  box2d
)
