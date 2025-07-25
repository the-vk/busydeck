set windows-shell := ["pwsh.exe", "-c"]

default: build

build: shaders
  cargo build

release: shaders
  cargo build --release

shaders:
  glslc shaders/shader.vert -o shaders/vert.spv
  glslc shaders/shader.frag -o shaders/frag.spv