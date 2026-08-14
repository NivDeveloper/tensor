cmake_minimum_required(VERSION 3.28)
project(tensor VERSION 0.1.0 LANGUAGES CXX)

# Header-only C++26 expression-template library. Requires a compiler with
# P2996 reflection — currently GCC 16+:
#
#   cmake -B build -DCMAKE_CXX_COMPILER=g++-16
#   cmake --build build && ctest --test-dir build
#
# Consumers: add_subdirectory / FetchContent and link tensor::tensor — or
# skip CMake entirely and add include/ to the include path with
# -std=c++26 -freflection.

add_library(tensor INTERFACE)
add_library(tensor::tensor ALIAS tensor)
target_include_directories(tensor INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(tensor INTERFACE cxx_std_26)

target_compile_options(tensor INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-freflection>)

# ── GPU evaluation (opt-in) ─────────────────────────────────────────────────
# Default OFF: the GPU path then does not exist at compile time — gpud is
# never fetched, tensor::gpu is never created, and including
# <Tensor/Gpu/Eval.h> is a #error naming this flag. ON fetches gpud (the
# five-operation device abstraction, pinned by tag) and adds tensor::gpu =
# tensor::tensor + gpud::gpud — the abstract interface only; the application
# links a backend (gpud::auto_, gpud::vulkan, …) itself. Not installed: gpud
# has no install rules — consume tensor::gpu via add_subdirectory/FetchContent.
option(TENSOR_ENABLE_GPU
    "GPU evaluation via gpud: fetch gpud and create tensor::gpu" OFF)
if(TENSOR_ENABLE_GPU)
    include(FetchContent)
    FetchContent_Declare(gpud
        GIT_REPOSITORY https://github.com/NivDeveloper/gpud.git
        # v0.2.0, pinned by hash so nothing moves between visits
        GIT_TAG        d73857fadf3cb5773c5cfb97afaff9b1fb47b674)
    FetchContent_MakeAvailable(gpud)

    add_library(tensor_gpu INTERFACE)
    add_library(tensor::gpu ALIAS tensor_gpu)
    target_link_libraries(tensor_gpu INTERFACE tensor gpud::gpud)
    target_compile_definitions(tensor_gpu INTERFACE TENSOR_GPU_ENABLED)
endif()

# ── map<f> GPU lowering via sdw (opt-in) ────────────────────────────────────
# Independent of TENSOR_ENABLE_GPU, matching the layer split: sdw is about
# codegen (detail/Gpu.h — translated function bodies in the emitted Slang),
# gpud about execution (Gpu/Eval.h). Unset: map<f> stays CPU-only and nothing
# changes anywhere. Set (path to a built sdw binary — github.com/NivDeveloper/
# shadow): the tensor_sdw_kernels() helper becomes available; apply it to any
# target with a kernels file. See docs/sdw-integration.md.
set(TENSOR_SDW_EXECUTABLE "" CACHE FILEPATH
    "Path to the sdw binary; enables GPU lowering of map<f> kernels")
if(TENSOR_SDW_EXECUTABLE AND
   EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorSdwKernels.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorSdwKernels.cmake)
endif()

if(PROJECT_IS_TOP_LEVEL)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
        message(FATAL_ERROR
            "tensor needs P2996 reflection (GCC 16+). Configure with "
            "-DCMAKE_CXX_COMPILER=g++-16 (found ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}).")
    endif()

    # The exact flags of every TU — what tools/ctime measures against, so
    # the compile-time harness never guesses an include path.
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    option(TENSOR_BUILD_TESTS "Build the tensor test suite" ON)
    option(TENSOR_BUILD_EXAMPLES "Build the tensor examples" ON)
    option(TENSOR_BUILD_BENCHMARKS
        "Build the tensor benchmarks (fetches Google Benchmark)" OFF)
    option(TENSOR_WERROR "Turn the warning set into errors (CI)" OFF)

    # Tests and examples build with the full warning set; the pinned CI
    # container adds -Werror. The INTERFACE target itself stays flag-free.
    set(TENSOR_WARNING_FLAGS -Wall -Wextra -Wpedantic)
    if(TENSOR_WERROR)
        list(APPEND TENSOR_WARNING_FLAGS -Werror)
    endif()

    # EXISTS as well as the option: the distributed tree carries the library
    # and the examples, so a missing development directory is a skip, not an
    # error.
    if(TENSOR_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        include(CTest)
        add_subdirectory(tests)
    endif()
    if(TENSOR_BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples)
        add_subdirectory(examples)
    endif()
    if(TENSOR_BUILD_BENCHMARKS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench)
        add_subdirectory(bench)
    endif()

    # Optional install: headers + a CMake package, so find_package(tensor)
    # works from an installed copy.
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS tensor EXPORT tensorTargets)
    install(EXPORT tensorTargets NAMESPACE tensor::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor)
    write_basic_package_version_file(tensorConfigVersion.cmake
        COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tensorConfig.cmake
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/tensorTargets.cmake\")\n")
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tensorConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tensorConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor)
endif()
