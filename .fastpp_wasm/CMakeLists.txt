cmake_minimum_required(VERSION 3.20)
project(fastpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fastpp.cpp)

# Default to Release when no build type is specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ─── WebAssembly release build (emcmake cmake ..) ─────────────────────────────
if(EMSCRIPTEN)

    add_executable(fastpp ${SOURCES})

    target_compile_options(fastpp PRIVATE -O3 -flto)

    target_link_options(fastpp PRIVATE
        -O3
        -flto
        -sWASM=1
        "-sEXPORTED_FUNCTIONS=[\"_fastpp\",\"_malloc\",\"_free\"]"
        "-sEXPORTED_RUNTIME_METHODS=[\"ccall\",\"cwrap\",\"HEAPU8\",\"HEAPF32\"]"
        -sALLOW_MEMORY_GROWTH=1
        -sNO_FILESYSTEM=1
        "-sENVIRONMENT=web,worker"
        -sMALLOC=emmalloc
        -sMODULARIZE=1
        "-sEXPORT_NAME=FastppModule"
    )

    # Emit fastpp.js + fastpp.wasm into build/
    set_target_properties(fastpp PROPERTIES SUFFIX ".js")

# ─── Native demo build (cmake ..) ─────────────────────────────────────────────
else()

    add_executable(fastpp_demo ${SOURCES})

    target_compile_definitions(fastpp_demo PRIVATE FASTPP_DEMO)

    target_compile_options(fastpp_demo PRIVATE -Wall -Wextra -Wpedantic)

endif()