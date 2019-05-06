cmake_minimum_required(VERSION 3.8.2)

project(quadrotor_model_codegen)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}")

message(STATUS ${CMAKE_MODULE_PATH})

message("\n\n### ${PROJECT_NAME}")
message("################################################################################")

find_package(ACADO REQUIRED)

add_executable(${PROJECT_NAME} quadrotor_model_thrustrates.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${ACADO_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC
        ${ACADO_SHARED_LIBRARIES} # Linux MacOS
        #${ACADO_STATIC_LIBRARIES} # Windows
        )

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
