# One directory per example — each is a standalone program with its own
# README.md and a two-line CMakeLists.txt (add_executable + link
# tensor::tensor), so an example folder is self-describing and copyable.
# An example with extra requirements (an opt-in dependency, a gating flag)
# keeps that logic inside its own directory.
#
# To add an example: create examples/<name>/ with <name>.cpp, README.md and
# a CMakeLists.txt, then register it here. See examples/README.md.

add_compile_options(${TENSOR_WARNING_FLAGS})

add_subdirectory(basic)
add_subdirectory(pendulum)
add_subdirectory(boltzmann)
add_subdirectory(mlp)
add_subdirectory(bgk)
add_subdirectory(gpu)
