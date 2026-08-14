# Needs the GPU path: configure with -DTENSOR_ENABLE_GPU=ON. The parent
# registers every example unconditionally; extra requirements gate here,
# inside the example — a CPU-only build just skips it.
if(NOT TARGET tensor::gpu)
    return()
endif()

add_executable(gpu gpu.cpp)
# gpud::auto_ provides open_default() and links in every backend that was
# built (Vulkan by default) — the app-side backend choice tensor never makes.
target_link_libraries(gpu PRIVATE tensor::gpu gpud::auto_)
