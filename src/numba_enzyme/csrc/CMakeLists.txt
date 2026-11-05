cmake_minimum_required(VERSION 3.20)
project(NumbaEnzyme LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_INSTALL_PREFIX}")

if(NOT DEFINED ENZYME_SOURCE_DIR)
  message(FATAL_ERROR "ENZYME_SOURCE_DIR must point to an Enzyme git checkout")
endif()

set(ENZYME_CLANG OFF CACHE BOOL "" FORCE)
set(ENZYME_MLIR OFF CACHE BOOL "" FORCE)
set(ENZYME_FORTRAN OFF CACHE BOOL "" FORCE)
set(ENZYME_FLANG OFF CACHE BOOL "" FORCE)
set(ENZYME_ENABLE_REACTANT OFF CACHE BOOL "" FORCE)
set(ENZYME_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ENZYME_STATIC_LIB ON CACHE BOOL "" FORCE)

add_subdirectory("${ENZYME_SOURCE_DIR}/enzyme" "${CMAKE_BINARY_DIR}/enzyme" EXCLUDE_FROM_ALL)

llvm_map_components_to_libnames(NUMBA_ENZYME_LLVM_LIBS
  Core
  IPO
  IRPrinter
  IRReader
  Linker
  Passes
  Support
)

add_library(NumbaEnzyme SHARED numba_enzyme_ffi.cpp)
set_target_properties(NumbaEnzyme PROPERTIES
  OUTPUT_NAME "NumbaEnzyme"
  PREFIX "lib"
)
target_include_directories(NumbaEnzyme PRIVATE ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(NumbaEnzyme PRIVATE ${LLVM_DEFINITIONS_LIST})

target_link_libraries(NumbaEnzyme PRIVATE
  "EnzymeStatic-${LLVM_VERSION_MAJOR}"
  ${NUMBA_ENZYME_LLVM_LIBS}
)

if(APPLE)
  target_link_options(NumbaEnzyme PRIVATE
    "-Wl,-dead_strip"
    "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/numba_enzyme.exp"
  )
elseif(UNIX)
  target_link_options(NumbaEnzyme PRIVATE
    "-Wl,--gc-sections"
    "-Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/numba_enzyme.lds"
  )
endif()
