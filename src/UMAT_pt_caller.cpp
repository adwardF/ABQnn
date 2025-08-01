#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <map>
#include <mutex>
#include <shared_mutex>

#include <ctime>

#include <torch/torch.h>
#include <torch/script.h> 

#include <windows.h>
#include <filesystem>

// Thread-safe module cache
std::map<std::string, torch::jit::Module> module_table;
std::shared_mutex module_table_mutex;  // Reader-writer lock for better performance

extern "C"
{
    __declspec(dllexport) int pt_module_invoke(char * module_file,
                          double *F, double *psi, double *Cauchy6, double *DDSDDE);
}

/*
Arguments:
F: double [3][3]

Returns:
psi: double
Cauchy6: double [6], use Voigt notation
DDSDDE: double [6][6]
*/

//#define DEBUG

__declspec(dllexport)  int pt_module_invoke(char *module_filename,
                                            double *F, double *psi,
                                            double *Cauchy6, double *DDSDDE)
{
    // Input validation
    if (!module_filename || !F || !psi || !Cauchy6 || !DDSDDE) {
        return 110;
    }

    std::string module_filename_str(module_filename); 

    #ifdef DEBUG
    static std::once_flag debug_flag;
    std::call_once(debug_flag, []() {
        freopen("D:/dev/ABQnn/pt_caller_err.txt", "w", stderr);   /* Replace with actual part */
    });
    #endif

    torch::jit::Module* mod_ptr = nullptr;
    
    // Fast path: Try to find module with shared (read) lock
    {
        std::shared_lock<std::shared_mutex> read_lock(module_table_mutex);
        auto it = module_table.find(module_filename_str);
        if (it != module_table.end()) {
            mod_ptr = &(it->second);  // Get pointer to avoid copy
        }
    }
    
    // Module not found, need to load it
    if (mod_ptr == nullptr) {
        std::unique_lock<std::shared_mutex> write_lock(module_table_mutex);
        
        // Double-check: another thread might have loaded it while we waited for write lock
        auto it = module_table.find(module_filename_str);
        if (it != module_table.end()) {
            mod_ptr = &(it->second);
        } else {
            // Load the module
            #ifdef DEBUG
            fprintf(stderr, "Module not found in table, loading: %s\n", module_filename);
            #endif
            try {
                torch::jit::Module module = torch::jit::load(module_filename_str, torch::kCPU);
                module.eval();  // Set to evaluation mode for better performance
                auto [inserted_it, success] = module_table.emplace(module_filename_str, std::move(module));
                if (success) {
                    mod_ptr = &(inserted_it->second);
                    #ifdef DEBUG
                    fprintf(stderr, "Module loaded and stored in table\n");
                    #endif
                } else {
                    #ifdef DEBUG
                    fprintf(stderr, "Failed to insert module into table\n");
                    #endif
                    return 102;
                }
            }
            catch(const std::exception& e) {
                #ifdef DEBUG
                fprintf(stderr, "Error loading module: %s\n", e.what());
                #endif
                return 101;
            }
        }
    }

    // Performance optimization: Create tensor directly in Float type to avoid conversion
    torch::Tensor F_tensor = torch::from_blob(F, {3, 3}, torch::kDouble)
                               .to(torch::kFloat)
                               .t()
                               .contiguous();  // Ensure contiguous memory layout
    
    torch::jit::IValue results;
    try {

        results = mod_ptr->forward({ F_tensor });
        
    }
    catch(const std::exception& e)
    {
        #ifdef DEBUG
        fprintf(stderr, "Error during model inference: %s\n", e.what());
        #endif
        return 105;
    }
    
    try {
        auto result_tuple = results.toTuple();

        double psi_value;
        auto &psi_result = result_tuple->elements()[0];
        if(psi_result.isDouble())
        {
            psi_value = psi_result.toDouble();
        }
        else if (psi_result.isTensor())
        {
            // Convert tensor to double
            auto psi_tensor = psi_result.toTensor().detach();
            if (psi_tensor.dtype() != torch::kDouble) {
                psi_tensor = psi_tensor.to(torch::kDouble);
            }
            psi_value = psi_tensor.item<double>();
        }
        else
        {
            #ifdef DEBUG
            fprintf(stderr, "Unexpected type for psi: %s\n", psi_result.tagKind());
            #endif
            return 106;
        }
        // More efficient tensor handling - avoid multiple conversions
        auto Cauchy6_tensor = result_tuple->elements()[1].toTensor().detach();
        auto DDSDDE_tensor = result_tuple->elements()[2].toTensor().detach();
        
        // Convert to double and copy in one step - more cache friendly
        if (Cauchy6_tensor.dtype() != torch::kDouble) {
            Cauchy6_tensor = Cauchy6_tensor.to(torch::kDouble);
        }
        if (DDSDDE_tensor.dtype() != torch::kDouble) {
            DDSDDE_tensor = DDSDDE_tensor.to(torch::kDouble);
        }
        
        // Ensure tensors are contiguous for efficient memcpy
        Cauchy6_tensor = Cauchy6_tensor.contiguous();
        DDSDDE_tensor = DDSDDE_tensor.contiguous();

        // Direct memory copy - most efficient
        *psi = psi_value;
        std::memcpy(Cauchy6, Cauchy6_tensor.data_ptr<double>(), 6 * sizeof(double));
        std::memcpy(DDSDDE, DDSDDE_tensor.data_ptr<double>(), 36 * sizeof(double));

    }
    catch(const std::exception& e)
    {
        #ifdef DEBUG
        fprintf(stderr, "Error during result conversion: %s\n", e.what());
        #endif
        return 105;
    }

    return 0;
}