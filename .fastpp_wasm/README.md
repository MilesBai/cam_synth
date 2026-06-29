

# to build

build local demo

```
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

build WASM

```
emcmake cmake -S . -B build_wasm
cmake --build build_wasm
```
