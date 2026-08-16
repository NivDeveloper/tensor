# The GPU path is optional: with -DTENSOR_ENABLE_GPU=ON the whole step
# loop runs as compute shaders, otherwise the identical expressions run on
# the CPU. Nothing about the physics changes.
add_executable(bgk bgk.cpp)
if(TARGET tensor::gpu)
    target_link_libraries(bgk PRIVATE tensor::gpu gpud::auto_)
else()
    target_link_libraries(bgk PRIVATE tensor::tensor)
endif()
