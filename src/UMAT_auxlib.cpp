#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <map>

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
static pt_module_invoke_func pt_module_invoke_handle = nullptr;

int try_load_dll(const char *dll_name)
{
    // First check if DLL is already loaded
    HMODULE existing_handle = GetModuleHandleA(dll_name);
    if (existing_handle != NULL)
    {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "DLL %s already loaded\n", dll_name);
        #endif
        return 0; // Already loaded
    }
    
    HMODULE handle = LoadLibraryExA(dll_name, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
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
    fprintf(stderr, "Successfully loaded %s\n", dll_name);
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

    // Set DLL search paths using configured LibTorch path
    if (!SetDllDirectoryA(ABQNN_LIBTORCH_LIB_PATH)) {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Warning: Failed to set DLL directory to %s (error=%lu)\n", 
                ABQNN_LIBTORCH_LIB_PATH, GetLastError());
        #endif
        return 1;
    } else {
        #ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Successfully set DLL directory to %s\n", ABQNN_LIBTORCH_LIB_PATH);
        #endif
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

    return 0;
}

int invoke_pt(const char *module_filename,
              const double *F, const double *mat_par, int n_mat_par,
              double *psi, double *Cauchy, double *DDSDDE)
{
    // Thread-safe, one-time initialization - no mutex overhead after first call
    std::call_once(init_flag, [&]() {
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
