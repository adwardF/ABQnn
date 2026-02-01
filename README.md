# ABQnn - PyTorch Neural Network Integration for Abaqus UMAT

A C++/Fortran library that enables using PyTorch neural network constitutive models within Abaqus finite element simulations through the UMAT (User Material) interface.

## Features

- **Neural Network Constitutive Models**: Use pre-trained TorchScript models for hyperelastic material response
- **Thread-Safe Caching**: Efficient model loading with reader-writer locks for parallel simulations
- **Fortran-C Interoperability**: Seamless integration with Abaqus UMAT via `iso_c_binding`
- **Cross-Platform**: Windows support with extensible design for Linux/macOS

## Project Structure

```
ABQnn/
├── CMakeLists.txt          # Root CMake configuration
├── cmake/                  # CMake modules
│   ├── FindLibTorch.cmake  # LibTorch finder
│   ├── config.h.in         # Config header template
│   └── ABQnnConfig.cmake.in
├── include/                # Public headers
│   ├── umat_pt_caller.h    # PyTorch caller API
│   └── umat_auxlib.h       # Auxiliary library API
├── src/                    # Source files
│   ├── CMakeLists.txt
│   ├── UMAT_pt_caller.cpp  # PyTorch model inference
│   └── UMAT_auxlib.cpp     # DLL loading and initialization
├── tests/                  # Test files
│   ├── CMakeLists.txt
│   ├── UMAT_fortest.f90    # Fortran test
│   └── pt_caller_test.cpp  # C++ test
├── models/                 # PyTorch models (.pt files)
├── fortran/                # UMAT Fortran files
│   └── UMAT_base.for       # Main UMAT subroutine
└── local/                  # Local development files
    └── allmodels/          # Pre-trained models
```

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
         -DLOG_PATH="D:/path/to/logs"
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_SHARED_LIBS` | ON | Build shared libraries |
| `BUILD_TESTS` | ON | Build test executables |
| `ENABLE_DEBUG_OUTPUT` | OFF | Enable debug logging to files |

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

### In Abaqus UMAT

```fortran
subroutine UMAT(STRESS, STATEV, DDSDDE, ...)
    use iso_c_binding
    
    interface
        function invoke_pt(ptname, F, psi, cauchy, C66) result(err) bind(C)
            ! ... interface definition
        end function
    end interface
    
    ! Call neural network model
    err = invoke_pt("path/to/model.pt"//c_null_char, DFGRD1, SSE, STRESS, DDSDDE)
end subroutine
```

### PyTorch Model Requirements

Your TorchScript model must accept a 3x3 deformation gradient tensor and return:
1. **psi** (scalar): Strain energy density
2. XXX **Cauchy6** (6-vector): Cauchy stress in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23]
3. XXX **DDSDDE** (6x6 matrix): Material tangent stiffness

## API Reference

### `pt_module_invoke`

```c
int pt_module_invoke(
    char* module_filename,  // Path to .pt file
    double* F,              // Deformation gradient [3x3]
    double* psi,            // Output: strain energy
    double* Cauchy,        // Output: Cauchy stress
    double* DDSDDE          // Output: tangent
);
```

### `invoke_pt`

Fortran-callable wrapper that handles DLL initialization.

## Error Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Failed to set DLL directory |
| 2 | Failed to load UMAT_pt_caller.dll |
| 3 | Failed to get DLL handle |
| 4 | Failed to get function address |
| 101 | Failed to load TorchScript module |
| 102 | Failed to cache module |
| 105 | Inference or conversion error |
| 106 | Unexpected output type |
| 110 | Invalid input (null pointer) |
