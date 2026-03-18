#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <map>
#include <filesystem>
#include <vector>

#include <ctime>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

#include "umat_auxlib.h"
#include "abqnn_config.h"

// Thread safety for initialization - most efficient approach
static std::once_flag init_flag;
static int initialization_error = 0;
typedef int (*pt_module_invoke_func)(const char*, const double*, const double*, int, double*, double*, double*);
typedef int (*pt_module_invoke_vumat_batch_func)(const char*, const double*, int, int, int, const double*, int, double*, double*);
static pt_module_invoke_func pt_module_invoke_handle = nullptr;
static pt_module_invoke_vumat_batch_func pt_module_invoke_vumat_batch_handle = nullptr;

static std::string get_loaded_module_path(HMODULE module)
{
    if (!module)
    {
        return std::string();
    }

    char path_buffer[MAX_PATH] = {0};
    DWORD n = GetModuleFileNameA(module, path_buffer, MAX_PATH);
    if (n == 0 || n >= MAX_PATH)
    {
        return std::string();
    }

    return std::string(path_buffer);
}

static void log_loaded_module(const char *module_name)
{
    HMODULE module = GetModuleHandleA(module_name);
    if (module == NULL)
    {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Module not loaded yet: %s\n", module_name);
        #endif
        return;
    }

    std::string loaded_path = get_loaded_module_path(module);
    #ifdef ENABLE_DEBUG_OUTPUT
    fprintf(stderr, "Module loaded: %s -> %s\n", module_name,
            loaded_path.empty() ? "<unknown>" : loaded_path.c_str());
    #endif
}

static int try_load_dll_absolute(const std::filesystem::path &dll_full_path)
{
    if (!std::filesystem::exists(dll_full_path))
    {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Absolute DLL path does not exist: %s\n", dll_full_path.string().c_str());
        #endif
        return 1;
    }

    HMODULE handle = LoadLibraryExA(dll_full_path.string().c_str(), NULL,
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (handle == NULL)
    {
        DWORD error = GetLastError();
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Failed to load %s: error code %lu", dll_full_path.string().c_str(), error);
        switch (error)
        {
            case 2: fprintf(stderr, " (File not found)\n"); break;
            case 3: fprintf(stderr, " (Path not found)\n"); break;
            case 126: fprintf(stderr, " (Module not found - missing dependencies)\n"); break;
            case 127: fprintf(stderr, " (Procedure not found - dependency issue)\n"); break;
            case 193: fprintf(stderr, " (Not a valid Win32 application)\n"); break;
            default: fprintf(stderr, " (Unknown error)\n"); break;
        }
        #endif
        return 1;
    }

    #ifdef ENABLE_DEBUG_OUTPUT
    std::string loaded_path = get_loaded_module_path(handle);
    fprintf(stderr, "Successfully loaded %s (resolved: %s)\n",
            dll_full_path.filename().string().c_str(),
            loaded_path.empty() ? "<unknown>" : loaded_path.c_str());
    #endif

    return 0;
}

int try_load_dll(const char *dll_name)
{
    HMODULE handle = LoadLibraryExA(dll_name, NULL,
                                    LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (handle == NULL)
    {
        DWORD error = GetLastError();
        if(error == ERROR_ALREADY_EXISTS) // 183
            return 0;
        
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Failed to load %s: error code %lu", dll_name, error);
        switch(error) {
            case 2: fprintf(stderr, " (File not found)\n"); break;
            case 3: fprintf(stderr, " (Path not found)\n"); break;
            case 126: fprintf(stderr, " (Module not found - missing dependencies)\n"); break;
            case 127: fprintf(stderr, " (Procedure not found - dependency issue)\n"); break;
            case 193: fprintf(stderr, " (Not a valid Win32 application)\n"); break;
            default: fprintf(stderr, " (Unknown error)\n"); break;
        }
        #endif
        return 1;
    }
    
    #ifdef ENABLE_DEBUG_OUTPUT
    std::string loaded_path = get_loaded_module_path(handle);
    fprintf(stderr, "Successfully loaded %s (resolved: %s)\n",
            dll_name, loaded_path.empty() ? "<unknown>" : loaded_path.c_str());
    #endif
    return 0;
}

int initialize_library()
{   
    // Note: This function is now called via std::call_once
    #ifdef ENABLE_DEBUG_OUTPUT
    
    char log_file_path[MAX_PATH];
    snprintf(log_file_path, MAX_PATH, "%s/auxlib_err.txt", ABQNN_LOG_PATH);
    freopen(log_file_path, "a", stderr);

    time_t now = time(NULL);
    fprintf(stderr, "UMAT_auxlib.cpp: %s", ctime(&now));
    fprintf(stderr, "Starting library initialization...\n");
    fprintf(stderr, "LibTorch lib path: %s\n", ABQNN_LIBTORCH_LIB_PATH);
    // Debug: Show current working directory
    char current_dir[MAX_PATH];
    if (GetCurrentDirectoryA(MAX_PATH, current_dir))
    {
        fprintf(stderr, "Current working directory: %s\n", current_dir);
    }
    #endif

    // Restrict default DLL search to user dirs + system32 to avoid Abaqus app-dir collisions
    if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32))
    {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Warning: SetDefaultDllDirectories failed (error=%lu)\n", GetLastError());
        #endif
    }

    // Set DLL search paths using configured LibTorch path
    // Convert ANSI string to wide string for AddDllDirectory
    wchar_t wide_path[MAX_PATH];
    mbstowcs(wide_path, ABQNN_LIBTORCH_LIB_PATH, MAX_PATH - 1);
    wide_path[MAX_PATH - 1] = L'\0';
    
    if (!AddDllDirectory(wide_path)) {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Warning: Failed to add DLL directory %s (error=%lu)\n", 
                ABQNN_LIBTORCH_LIB_PATH, GetLastError());
        #endif
        return 1;
    } else {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Successfully added DLL directory %s\n", ABQNN_LIBTORCH_LIB_PATH);
        #endif
    }

    // Preload CRT from System32 early to reduce risk of older Abaqus-bundled CRT collisions
    char system_dir[MAX_PATH] = {0};
    UINT n_system = GetSystemDirectoryA(system_dir, MAX_PATH);
    if (n_system > 0 && n_system < MAX_PATH)
    {
        static const char* kCrtDlls[] = {
            "msvcp140.dll",
            "vcruntime140.dll",
            "vcruntime140_1.dll",
        };

        std::filesystem::path system_dir_path(system_dir);
        for (const char *dll_name : kCrtDlls)
        {
            std::filesystem::path dll_path = system_dir_path / dll_name;
            try_load_dll_absolute(dll_path);
        }
    }

    #ifdef ENABLE_DEBUG_OUTPUT
    log_loaded_module("msvcp140.dll");
    log_loaded_module("vcruntime140.dll");
    log_loaded_module("vcruntime140_1.dll");
    log_loaded_module("libiomp5md.dll");
    #endif

    // Preload critical LibTorch dependencies by absolute path to avoid wrong-version DLL resolution
    static const char* kTorchDlls[] = {
        "torch_global_deps.dll",
        "libiomp5md.dll",
        "cupti64_2025.3.0.dll",
        "c10.dll",
        "torch_cpu.dll",
    };

    std::filesystem::path torch_lib_dir(ABQNN_LIBTORCH_LIB_PATH);
    for (const char *dll_name : kTorchDlls)
    {
        std::filesystem::path dll_path = torch_lib_dir / dll_name;
        try_load_dll_absolute(dll_path);
    }

    /* try_load_dll("torch_cuda.dll"); */
    
    int ret = try_load_dll("UMAT_pt_caller.dll");
    if(ret)
    {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Failed to load UMAT_pt_caller.dll (ret=%d)\n", ret);
        #endif
        return 2;
    }

    HMODULE dll = GetModuleHandleA("UMAT_pt_caller.dll");
    if(dll == NULL)
    {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Failed to get handle for UMAT_pt_caller.dll (error=%lu)\n", GetLastError());
        #endif
        return 3;
    }

    pt_module_invoke_handle = (pt_module_invoke_func)GetProcAddress(dll, "pt_module_invoke");
    if(pt_module_invoke_handle == NULL)
    {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Failed to get function address for pt_module_invoke (error=%lu)\n", GetLastError());
        #endif
        return 4;
    }

    pt_module_invoke_vumat_batch_handle = (pt_module_invoke_vumat_batch_func)GetProcAddress(dll, "pt_module_invoke_vumat_batch");
    if(pt_module_invoke_vumat_batch_handle == NULL)
    {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Failed to get function address for pt_module_invoke_vumat_batch (error=%lu)\n", GetLastError());
        #endif
        return 4;
    }

    return 0;
}

int invoke_pt(const char *module_filename,
              const double *F, const double *mat_par, int n_mat_par,
              double *psi, double *Cauchy, double *DDSDDE)
{
    // Thread-safe, one-time initialization - no mutex overhead after first call
    std::call_once(init_flag, []() {
        initialization_error = initialize_library();
    });
    
    // Check if initialization failed
    if (initialization_error != 0)
    {
        return initialization_error;
    }
    
    // Call the C function from UMAT_pt_caller.dll
    int err = pt_module_invoke_handle(module_filename, F, mat_par, n_mat_par, psi, Cauchy, DDSDDE);
    
    return err;
}

int invoke_pt_vumat_batch(const char *module_filename,
                          const double *defgradF,
                          int nblock,
                          int ndir,
                          int nshr,
                          const double *mat_par,
                          int n_mat_par,
                          double *enerInternNew,
                          double *stressNew)
{
    std::call_once(init_flag, []() {
        initialization_error = initialize_library();
    });

    if (initialization_error != 0)
    {
        return initialization_error;
    }

    int err = pt_module_invoke_vumat_batch_handle(module_filename,
                                                  defgradF,
                                                  nblock,
                                                  ndir,
                                                  nshr,
                                                  mat_par,
                                                  n_mat_par,
                                                  enerInternNew,
                                                  stressNew);

    return err;
}