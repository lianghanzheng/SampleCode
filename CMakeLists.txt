cmake_minimum_required(VERSION 3.20.0)

project("sysy-opt" LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#add_definitions("-fno-rtti")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_CURRENT_BINARY_DIR}/include)
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(tools)

link_libraries(${LLVM_BUILD_LIBRARY_DIR})