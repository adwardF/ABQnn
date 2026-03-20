# ABQnn - PyTorch Neural Network Integration for Abaqus UMAT

A C++/Fortran library that enables using PyTorch neural network constitutive models within Abaqus finite element simulations through UMAT and VUMAT interfaces.

## Features

- **Neural Network Constitutive Models**: Use pre-trained TorchScript models for hyperelastic material response
- **Out-of-Process Inference**: Named-pipe IPC between Abaqus-facing client (`umat_auxlib`) and Torch server (`abqnn_inference_server`)
- **Thread-Safe Caching**: Efficient server-side model caching with reader-writer locks for parallel simulations
- **Fortran-C Interoperability**: Seamless integration with Abaqus UMAT via `iso_c_binding`
- **Windows Platform**: Current implementation targets Windows only

## Project Structure

```
ABQnn/
├── CMakeLists.txt          # Root CMake configuration
├── cmake/                  # CMake modules
│   ├── FindLibTorch.cmake  # LibTorch finder
│   ├── config.h.in         # Config header template
│   └── ABQnnConfig.cmake.in
├── include/                # Public headers
│   ├── abqnn_ipc_common.h  # IPC helpers/shared protocol utilities
│   ├── abqnn_ipc_protocol.h# IPC protocol constants
│   └── umat_auxlib.h       # Auxiliary library API
├── src/                    # Source files
│   ├── CMakeLists.txt
│   ├── ABQnn_inference_server.cpp # Torch inference server
│   ├── abqnn_ipc_common.cpp       # IPC implementation
│   └── UMAT_auxlib.cpp            # Abaqus-facing IPC client
├── tests/                  # Test files
│   ├── CMakeLists.txt
│   ├── UMAT_fortest.f90    # Fortran test
│   ├── VUMAT_fortest.f90   # VUMAT Fortran test
│   └── pt_caller_test.cpp  # C++ IPC client test
├── models/                 # PyTorch models (.pt files)
├── fortran/                # UMAT Fortran files
│   ├── UMAT_base.for       # Main UMAT subroutine
│   └── VUMAT_base.for      # Main VUMAT subroutine
└── local/                  # Local development files
    └── allmodels/          # Pre-trained models
```

## Platform Support

- **Windows only** is supported at this time.
- The IPC transport is implemented with Win32 Named Pipes (`\\.\pipe\abqnn_inference`).
- CMake now fails early on non-Windows platforms.

## Requirements

- **CMake** >= 3.18
- **C++ Compiler** with C++17 support (MSVC, GCC, Clang)
- **LibTorch** (PyTorch C++ distribution)
- **Fortran Compiler** (Intel ifx or gfortran) - for tests
- **Abaqus** 2021+ (for deployment)

## Building

### Configure

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake (all paths are configurable)
cmake .. -DLIBTORCH_PATH="D:/Library/libtorch" \
         -DABAQUS_LIB_PATH="D:/SIMULIA/EstProducts/2021/win_b64/code/lib" \
         -DMODEL_PATH="D:/path/to/models" \
         -DLOG_PATH="D:/path/to/logs" \
         -DABQNN_UMAT_TORCH_DEVICE=CPU \
         -DABQNN_VUMAT_TORCH_DEVICE=CUDA
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_SHARED_LIBS` | ON | Build shared libraries |
| `BUILD_TESTS` | ON | Build test executables |
| `ENABLE_DEBUG_OUTPUT` | OFF | Enable debug logging to files |
| `ABQNN_UMAT_TORCH_DEVICE` | CPU | UMAT inference device (`CPU` or `CUDA`) |
| `ABQNN_VUMAT_TORCH_DEVICE` | CPU | VUMAT inference device (`CPU` or `CUDA`) |

### Configurable Paths

All paths are configured at compile time and embedded into the binaries:

| Variable | Default | Description |
|----------|---------|-------------|
| `LIBTORCH_PATH` | Auto-detected | Path to LibTorch installation |
| `LIBTORCH_LIB_PATH` | `${LIBTORCH_PATH}/lib` | LibTorch DLL directory (runtime) |
| `ABAQUS_LIB_PATH` | Platform-specific | Path to Abaqus library directory |
| `MODEL_PATH` | `${PROJECT}/models` | Default path for PyTorch models |
| `LOG_PATH` | `${PROJECT}/log` | Debug log output directory |

These paths are written to `abqnn_config.h` during CMake configuration.

### Compile

```bash
# Build
cmake --build . --config Release

# Install
cmake --install . --config Release
```

### Visual Studio

```powershell
# Configure for Visual Studio
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build
cmake --build . --config Release
```

## Usage

### Runtime Architecture

1. Start `abqnn_inference_server` (the out-of-process TorchScript server).
2. Abaqus UMAT/VUMAT calls `invoke_pt` / `invoke_pt_vumat_batch` from `umat_auxlib`.
3. `umat_auxlib` sends requests over `\\.\pipe\abqnn_inference`.
4. Server loads/caches model and returns inference outputs.

### In Abaqus UMAT

```fortran
subroutine UMAT(STRESS, STATEV, DDSDDE, ...)
    use iso_c_binding
    
    interface
        function invoke_pt(ptname, F, mat_par, n_mat_par, psi, cauchy, C66) result(err) bind(C)
            ! ... interface definition
        end function
    end interface
    
    ! Call neural network model
    err = invoke_pt("path/to/model.pt"//c_null_char, DFGRD1, PROPS, NPROPS, SSE, STRESS, DDSDDE)
end subroutine
```

### PyTorch Model Requirements

Your TorchScript model must accept a 3x3 deformation gradient tensor and return:
1. **psi** (scalar): Strain energy density
2. XXX **Cauchy6** (6-vector): Cauchy stress in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23]
3. XXX **DDSDDE** (6x6 matrix): Material tangent stiffness

## API Reference

### `invoke_pt`

```c
int invoke_pt(
    const char* module_filename,
    const double* F,
    const double* mat_par,
    int n_mat_par,
    double* psi,
    double* Cauchy,
    double* DDSDDE
);
```

### `invoke_pt_vumat_batch`

```c
int invoke_pt_vumat_batch(
    const char* module_filename,
    const double* defgradF,
    int nblock,
    int ndir,
    int nshr,
    const double* mat_par,
    int n_mat_par,
    double* enerInternNew,
    double* stressNew
);
```

For 3D VUMAT (`ndir=3`, `nshr=3`), deformation gradient component ordering is
`[F11, F22, F33, F12, F23, F31, F21, F32, F13]`.

Notes:
- A server process must be running and reachable on `\\.\pipe\abqnn_inference`.
- `n_mat_par` must be non-negative.
- If `n_mat_par > 0`, `mat_par` must be non-null.

## Error Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 101 | Failed to load TorchScript model on server |
| 102 | Failed to cache model on server |
| 105 | Inference or conversion error |
| 106 | Unexpected scalar output type |
| 110 | Invalid input arguments |
| 111 | Unsupported tensor layout/shape mismatch |
| 112 | CUDA requested but unavailable |
| 120 | IPC connect error |
| 121 | IPC write error |
| 122 | IPC read error |
| 123 | IPC protocol error |
