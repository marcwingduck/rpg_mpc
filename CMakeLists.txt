cmake_minimum_required(VERSION 3.8.2)

project(mpc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ##############################################################################

add_library(mpc_solver
        externals/qpoases/SRC/Bounds.cpp
        externals/qpoases/SRC/Constraints.cpp
        externals/qpoases/SRC/CyclingManager.cpp
        externals/qpoases/SRC/Indexlist.cpp
        externals/qpoases/SRC/MessageHandling.cpp
        externals/qpoases/SRC/QProblem.cpp
        externals/qpoases/SRC/QProblemB.cpp
        externals/qpoases/SRC/SubjectTo.cpp
        externals/qpoases/SRC/Utils.cpp
        externals/qpoases/SRC/EXTRAS/SolutionAnalysis.cpp
        model/quadrotor_mpc_codegen/acado_qpoases_interface.cpp
        model/quadrotor_mpc_codegen/acado_integrator.c
        model/quadrotor_mpc_codegen/acado_solver.c
        model/quadrotor_mpc_codegen/acado_auxiliary_functions.c
        )

target_include_directories(mpc_solver PUBLIC
        model/quadrotor_mpc_codegen/
        externals/qpoases
        externals/qpoases/INCLUDE
        externals/qpoases/SRC
        )

# ##############################################################################

add_library(mpc_wrapper
        include/rpg_mpc/mpc_params.h
        include/rpg_mpc/mpc_wrapper.h
        src/mpc_wrapper.cpp
        )

target_include_directories(mpc_wrapper PUBLIC
        include
        )

target_link_libraries(mpc_wrapper
        Eigen3::Eigen
        mpc_solver
        )

# ##############################################################################

add_library(mpc_controller
        include/rpg_mpc/mpc_controller.h
        src/mpc_controller.cpp
        )

target_include_directories(mpc_controller PUBLIC
        include
        )

target_link_libraries(mpc_controller
        utils
        mpc_wrapper
        )