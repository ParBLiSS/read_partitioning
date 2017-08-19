cmake_minimum_required(VERSION 2.6)

# project settings
project(new_readpartitioning)

##### General Compilation Settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11  -Drestrict=__restrict__  -ffast-math -DGRAPH_GENERATOR_MPI -DGRAPHGEN_DISTRIBUTED_MEMORY")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -funroll-loops")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -funroll-loops")

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
  set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH} /lib /usr/lib")
endif()

#Set default cmake build type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)


if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

#### MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
  set(EXTRA_LIBS ${EXTRA_LIBS} ${MPI_LIBRARIES})
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
  message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

###### Executable and Libraries
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

# include external repositories : mxx, graph500-gen, bliss 
#include_directories("${PROJECT_SOURCE_DIR}/ext/mxx/include")
#include_directories("${PROJECT_SOURCE_DIR}/ext/mxx/ext")
include_directories("${PROJECT_SOURCE_DIR}/ext")

#BLISS
add_subdirectory("${PROJECT_SOURCE_DIR}/ext/bliss")   #BLISS cmake needs to be executed
include_directories("${PROJECT_SOURCE_DIR}/ext/bliss/src/")
include_directories("${PROJECT_SOURCE_DIR}/ext/bliss/ext/")
include_directories("${PROJECT_SOURCE_DIR}/ext/bliss/ext/mxx/include")
include_directories("${PROJECT_SOURCE_DIR}/ext/bliss/ext/mxx/ext")
include_directories("${PROJECT_SOURCE_DIR}/ext/bliss/ext/tclap-1.2.1/include")
include_directories("${PROJECT_SOURCE_DIR}/ext/bliss/ext/Nadeau")
include_directories("${PROJECT_SOURCE_DIR}/ext/bliss/ext/sparsehash/src")
include_directories("${PROJECT_SOURCE_DIR}/ext/bliss/test/test")


#include our build directory for bliss config files
include_directories("${CMAKE_BINARY_DIR}")


# include our src files
include_directories("${PROJECT_SOURCE_DIR}/src")
add_executable(dbg_partition
  src/utils/io_utils.cpp
  src/dbg_partition.cpp)
target_link_libraries(dbg_partition ${EXTRA_LIBS})


add_executable(construct_plain_dbg
  src/construct_plain_dbg.cpp)
target_link_libraries(construct_plain_dbg ${EXTRA_LIBS})

add_executable(eval_partition
  src/utils/io_utils.cpp
  src/eval_partition.cpp)
target_link_libraries(eval_partition ${EXTRA_LIBS})

## add gtest
#include_directories("${PROJECT_SOURCE_DIR}/gtest")
#add_subdirectory("${PROJECT_SOURCE_DIR}/gtest")

## add own subdirectories
#add_subdirectory("${PROJECT_SOURCE_DIR}/test")
