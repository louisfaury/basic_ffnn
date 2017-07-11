# defining project
cmake_minimum_required (VERSION 2.8)
project (ffnn)
set (PATH ${CMAKE_CURRENT_SOURCE_DIR})

# include
INCLUDE_DIRECTORIES(PATH ../lib/)

# add subdirectories
add_subdirectory(dataset/)
add_subdirectory(neuron/)
add_subdirectory(layer/)
add_subdirectory(neural_network/)
add_subdirectory(neural_trainer/)

# adds files
set(SRCS
    ${SRCS}
    main.cpp
    )

#

# adds executable to binary tree
add_executable(main ${SRCS})
target_compile_features(main PRIVATE cxx_range_for)


