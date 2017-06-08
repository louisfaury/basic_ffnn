# defining project
cmake_minimum_required (VERSION 2.8)
project (ffnn)
set (PATH ${CMAKE_CURRENT_SOURCE_DIR})

# include
INCLUDE_DIRECTORIES(PATH)

# adds files
set(SRCS
    main.cpp
    Neuron.cpp
    SigmoidNeuron.cpp
    LinearNeuron.cpp
    Layer.cpp
    SigmoidLayer.cpp
    LinearLayer.cpp
    )

# adds executable to binary tree
add_executable(main ${SRCS})
target_compile_features(main PRIVATE cxx_range_for)


