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

#ifdef _WIN32
#include <windows.h>
#endif

#include <filesystem>

#include "umat_pt_caller.h"
#include "abqnn_config.h"

// Thread-safe module cache
static std::map<std::string, torch::jit::Module> module_table;
static std::shared_mutex module_table_mutex; // Reader-writer lock for better performance

int try_load_module(const char *module_filename,
                    torch::jit::Module *&out_module);

int readout_results(
    const torch::jit::IValue &results,
    double *psi,
    double *Cauchy,
    double *DDSDDE);

/*
Arguments:
F: double [3][3]
mat_par: double [n_mat_par]

Returns:
psi: double
Cauchy: double, use Voigt notation
DDSDDE: double
*/

UMAT_PT_CALLER_API int pt_module_invoke(const char *module_filename,
                                        const double *F,
                                        const double *mat_par,
                                        int n_mat_par,
                                        double *psi,
                                        double *Cauchy, double *DDSDDE)
{
    if (!module_filename || !F || !psi || !Cauchy || !DDSDDE)
    {
        return 110;
    }

    std::string module_filename_str(module_filename);

#ifdef ENABLE_DEBUG_OUTPUT
    static std::once_flag debug_flag;
    std::call_once(debug_flag, []()
                   {
        static std::string log_file_path = (std::filesystem::path(ABQNN_LOG_PATH) / "pt_caller_err.txt").string();
        freopen(log_file_path.c_str(), "a", stderr);
        time_t now = time(NULL);
        fprintf(stderr, "UMAT_pt_caller.cpp: %s", ctime(&now)); });
#endif

    torch::jit::Module *mod_ptr = nullptr;
    int mod_load_err = try_load_module(module_filename, mod_ptr);

    if (mod_load_err != 0)
    {
        return mod_load_err; // Propagate loading error
    }
    torch::Tensor F_tensor = torch::from_blob((void*)F, {3, 3}, torch::kDouble)
                                 .t()
                                 .contiguous(); // Ensure contiguous memory layout

    torch::Tensor mat_par_tensor;

    if (mat_par && n_mat_par > 0)
    {
        mat_par_tensor = torch::from_blob((void*)mat_par, {n_mat_par}, torch::kDouble).contiguous();
    }
    else
    {
        mat_par_tensor = torch::empty({0}, torch::kDouble);
    }

    torch::jit::IValue results;
    try
    {

        results = mod_ptr->forward({F_tensor, mat_par_tensor});
    }
    catch (const std::exception &e)
    {
#ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Error during model inference: %s\n", e.what());
#endif
        return 105;
    }

    int readout_err = readout_results(results, psi, Cauchy, DDSDDE);
    if (readout_err != 0)
    {
        return readout_err;
    }

    return 0;
}

int try_load_module(const char *module_filename,
                    torch::jit::Module *&out_module)
{
    if (!module_filename)
    {
        return 110;
    }

    std::string module_filename_str(module_filename);

    bool is_loaded = false;
    {
        std::shared_lock<std::shared_mutex> read_lock(module_table_mutex);
        auto it = module_table.find(module_filename_str);
        if (it != module_table.end())
        {
            out_module = &(it->second);
            is_loaded = true;
        }
    }
    if (is_loaded)
    {
        return 0; // Module found in cache
    }
    else
    {
        // Module not found in cache, load it
        std::unique_lock<std::shared_mutex> write_lock(module_table_mutex);

        // Double-check: another thread might have loaded it while we waited for write lock
        auto it = module_table.find(module_filename_str);
        if (it != module_table.end())
        {
            out_module = &(it->second);
        }
        else
        {
#ifdef ENABLE_DEBUG_OUTPUT
            fprintf(stderr, "Module not found in table, loading: %s\n", module_filename);
#endif
            try
            {
                std::filesystem::current_path(ABQNN_MODEL_PATH);
#ifdef ENABLE_DEBUG_OUTPUT
                fprintf(stderr, "Changed working directory to: %s\n", std::filesystem::current_path().string().c_str());
                fprintf(stderr, "Trying to load: %s\n", module_filename_str.c_str());
#endif
                torch::jit::Module module = torch::jit::load(module_filename_str, torch::kCPU);
                module.to(torch::kCPU);
                module.eval(); // Set to evaluation mode for better performance
#ifdef ENABLE_DEBUG_OUTPUT
                fprintf(stderr, "Loaded: %s\n", module_filename_str.c_str());
#endif
                auto [inserted_it, success] = module_table.emplace(module_filename_str, std::move(module));
                if (success)
                {
                    out_module = &(inserted_it->second);
#ifdef ENABLE_DEBUG_OUTPUT
                    fprintf(stderr, "Module loaded and stored in table\n");
#endif
                    return 0;
                }
                else
                {
#ifdef ENABLE_DEBUG_OUTPUT
                    fprintf(stderr, "Failed to insert module into table\n");
#endif
                    return 102;
                }
            }
            catch (const std::exception &e)
            {
#ifdef ENABLE_DEBUG_OUTPUT
                fprintf(stderr, "Error loading module: %s\n", e.what());
#endif
                return 101;
            }
        }
    }
}

int readout_results(
    const torch::jit::IValue &results,
    double *psi,
    double *Cauchy,
    double *DDSDDE)
{
    try
    {
        auto result_tuple = results.toTuple();

        double psi_value;
        auto &psi_result = result_tuple->elements()[0];
        if (psi_result.isDouble())
        {
            psi_value = psi_result.toDouble();
        }
        else if (psi_result.isTensor())
        {
            // Convert tensor to double
            auto psi_tensor = psi_result.toTensor().detach();
            if (psi_tensor.dtype() != torch::kDouble)
            {
                psi_tensor = psi_tensor.to(torch::kDouble);
            }
            if (psi_tensor.device() != torch::kCPU)
            {
                psi_tensor = psi_tensor.to(torch::kCPU);
            }
            psi_value = psi_tensor.item<double>();
        }
        else
        {
#ifdef ENABLE_DEBUG_OUTPUT
            fprintf(stderr, "Unexpected type for psi: %s\n", psi_result.tagKind().c_str());
#endif
            return 106;
        }
        // More efficient tensor handling - avoid multiple conversions
        auto Cauchy_tensor = result_tuple->elements()[1].toTensor().detach();
        auto DDSDDE_tensor = result_tuple->elements()[2].toTensor().detach();

        // Convert to double and copy in one step - more cache friendly
        if (Cauchy_tensor.dtype() != torch::kDouble)
        {
            Cauchy_tensor = Cauchy_tensor.to(torch::kDouble);
        }
        if (Cauchy_tensor.device() != torch::kCPU)
        {
            Cauchy_tensor = Cauchy_tensor.to(torch::kCPU);
        }
        if (DDSDDE_tensor.dtype() != torch::kDouble)
        {
            DDSDDE_tensor = DDSDDE_tensor.to(torch::kDouble);
        }
        if (DDSDDE_tensor.device() != torch::kCPU)
        {
            DDSDDE_tensor = DDSDDE_tensor.to(torch::kCPU);
        }

        // Ensure tensors are contiguous for memcpy
        Cauchy_tensor = Cauchy_tensor.contiguous();
        DDSDDE_tensor = DDSDDE_tensor.contiguous();

        size_t Cauchy_size = Cauchy_tensor.numel();
        size_t DDSDDE_size = DDSDDE_tensor.numel();

        *psi = psi_value;
        std::memcpy(Cauchy, Cauchy_tensor.data_ptr<double>(), Cauchy_size * sizeof(double));
        std::memcpy(DDSDDE, DDSDDE_tensor.data_ptr<double>(), DDSDDE_size * sizeof(double));
    }
    catch (const std::exception &e)
    {
#ifdef ENABLE_DEBUG_OUTPUT
        fprintf(stderr, "Error during result conversion: %s\n", e.what());
#endif
        return 105;
    }
    return 0;
}
