add_executable(pendulum pendulum.cpp)
target_link_libraries(pendulum PRIVATE tensor::tensor)
