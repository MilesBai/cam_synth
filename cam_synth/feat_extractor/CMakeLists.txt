cmake_minimum_required(VERSION 3.18)
project(feat_extractor CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(_feat_extractor
    bindings.cpp
    feat_extractor.cpp
)

target_include_directories(_feat_extractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install the .so next to the Python package's __init__.py
install(TARGETS _feat_extractor
    LIBRARY DESTINATION cam_synth/feat_extractor
)
