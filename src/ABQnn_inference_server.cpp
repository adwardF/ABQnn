#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <filesystem>
#include <thread>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "abqnn_config.h"
#include "abqnn_ipc_protocol.h"
#include "abqnn_ipc_common.h"

static std::map<std::string, torch::jit::Module> module_table;
static std::shared_mutex module_table_mutex;

enum class RequestKind
{
    UMAT,
    VUMAT
};

static const char *get_configured_device_name(RequestKind request_kind)
{
    return request_kind == RequestKind::UMAT ? ABQNN_UMAT_TORCH_DEVICE : ABQNN_VUMAT_TORCH_DEVICE;
}

static torch::Device get_inference_device(RequestKind request_kind)
{
    if (std::strcmp(get_configured_device_name(request_kind), "CUDA") == 0)
    {
        if (torch::cuda::is_available())
        {
            return torch::Device(torch::kCUDA);
        }
        return torch::Device(torch::kCPU);
    }
    return torch::Device(torch::kCPU);
}

static int validate_inference_devices()
{
    const bool umat_wants_cuda = std::strcmp(ABQNN_UMAT_TORCH_DEVICE, "CUDA") == 0;
    const bool vumat_wants_cuda = std::strcmp(ABQNN_VUMAT_TORCH_DEVICE, "CUDA") == 0;

    if ((umat_wants_cuda || vumat_wants_cuda))
    {
        // Due to some reasons, we need to first load torch_cuda.dll manually 
        // before any CUDA-related API is called
        std::wstring dll_path_w = std::filesystem::path(ABQNN_LIBTORCH_LIB_PATH).wstring();
        LPCWSTR dll_path = dll_path_w.c_str();
        if(!AddDllDirectory(dll_path))
        {
            DWORD err = GetLastError();
#ifdef ENABLE_DEBUG_OUTPUT
            std::fprintf(stderr, "server: warning: failed to add %ls to DLL search path (%lu), CUDA inference may not work\n", dll_path, err);
#endif
        }
        if(!LoadLibraryExA("torch_cuda.dll", NULL, LOAD_LIBRARY_SEARCH_USER_DIRS))
        {
            DWORD err = GetLastError();
#ifdef ENABLE_DEBUG_OUTPUT
            std::fprintf(stderr, "server: warning: failed to load torch_cuda.dll (%lu), CUDA inference will not work\n", err);
#endif
        }
        else
        {
#ifdef ENABLE_DEBUG_OUTPUT
            std::fprintf(stderr, "server: successfully loaded torch_cuda.dll\n");
#endif
        }

        if(torch::cuda::is_available())
        {
            return 0;
        }
        else
        {
#ifdef ENABLE_DEBUG_OUTPUT
            std::fprintf(stderr, "server: CUDA requested but CUDA is not available\n");
#endif
            return 112;
        }
    }
    return 0;
}

static int try_load_module(const char *module_filename, RequestKind request_kind, torch::jit::Module *&out_module)
{
    std::string module_filename_str(module_filename);
    std::string module_cache_key = module_filename_str + "|" + get_configured_device_name(request_kind);

    {
        std::shared_lock<std::shared_mutex> lock(module_table_mutex);
        auto it = module_table.find(module_cache_key);
        if (it != module_table.end())
        {
            out_module = &it->second;
            return 0;
        }
    }

    std::unique_lock<std::shared_mutex> lock(module_table_mutex);
    auto it = module_table.find(module_cache_key);
    if (it != module_table.end())
    {
        out_module = &it->second;
        return 0;
    }

    try
    {
        std::filesystem::current_path(ABQNN_MODEL_PATH);
        auto inference_device = get_inference_device(request_kind);
        torch::jit::Module module = torch::jit::load(module_filename_str, inference_device);
        module.to(inference_device);
        module.eval();

        auto [inserted_it, success] = module_table.emplace(module_cache_key, std::move(module));
        if (!success)
        {
            return 102;
        }
        out_module = &inserted_it->second;
        return 0;
    }
    catch (const std::exception &e)
    {
#ifdef ENABLE_DEBUG_OUTPUT
        std::fprintf(stderr, "server: model load failed: %s\n", e.what());
#endif
        return 101;
    }
}

static int build_defgrad_batch_tensor(const double *defgradF,
                                      int nblock,
                                      int ndir,
                                      int nshr,
                                      torch::Tensor &F_batch_tensor)
{
    if (!defgradF || nblock <= 0)
    {
        return 110;
    }

    const int ndefgrad = ndir + 2 * nshr;
    if (!((ndir == 3 && nshr == 3 && ndefgrad == 9) || (ndir == 3 && nshr == 1 && ndefgrad == 5)))
    {
        return 111;
    }

    auto defgrad_fortran = torch::from_blob((void *)defgradF, {ndefgrad, nblock}, torch::kDouble).t().contiguous();
    auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);
    F_batch_tensor = torch::zeros({nblock, 3, 3}, options);

    F_batch_tensor.index_put_({torch::indexing::Slice(), 0, 0}, defgrad_fortran.index({torch::indexing::Slice(), 0}));
    F_batch_tensor.index_put_({torch::indexing::Slice(), 1, 1}, defgrad_fortran.index({torch::indexing::Slice(), 1}));
    F_batch_tensor.index_put_({torch::indexing::Slice(), 2, 2}, defgrad_fortran.index({torch::indexing::Slice(), 2}));
    F_batch_tensor.index_put_({torch::indexing::Slice(), 0, 1}, defgrad_fortran.index({torch::indexing::Slice(), 3}));
    F_batch_tensor.index_put_({torch::indexing::Slice(), 1, 0}, defgrad_fortran.index({torch::indexing::Slice(), (ndir == 3 && nshr == 3) ? 6 : 4}));

    if (ndir == 3 && nshr == 3)
    {
        F_batch_tensor.index_put_({torch::indexing::Slice(), 1, 2}, defgrad_fortran.index({torch::indexing::Slice(), 4}));
        F_batch_tensor.index_put_({torch::indexing::Slice(), 2, 0}, defgrad_fortran.index({torch::indexing::Slice(), 5}));
        F_batch_tensor.index_put_({torch::indexing::Slice(), 2, 1}, defgrad_fortran.index({torch::indexing::Slice(), 7}));
        F_batch_tensor.index_put_({torch::indexing::Slice(), 0, 2}, defgrad_fortran.index({torch::indexing::Slice(), 8}));
    }

    return 0;
}

static int decode_umat_results(const torch::jit::IValue &results,
                               double &psi,
                               std::vector<double> &cauchy,
                               std::vector<double> &ddsdde)
{
    if (!results.isTuple())
    {
        return 111;
    }

    auto result_tuple = results.toTuple();
    const auto &elements = result_tuple->elements();
    if (elements.size() < 3)
    {
        return 111;
    }

    const auto &psi_result = elements[0];
    if (psi_result.isDouble())
    {
        psi = psi_result.toDouble();
    }
    else if (psi_result.isTensor())
    {
        auto psi_tensor = psi_result.toTensor().to(torch::kCPU).to(torch::kDouble);
        if (psi_tensor.numel() != 1)
        {
            return 111;
        }
        psi = psi_tensor.item<double>();
    }
    else
    {
        return 106;
    }

    if (!elements[1].isTensor() || !elements[2].isTensor())
    {
        return 111;
    }

    auto cauchy_tensor = elements[1].toTensor().to(torch::kCPU).to(torch::kDouble).contiguous().reshape({-1});
    auto ddsdde_tensor = elements[2].toTensor().to(torch::kCPU).to(torch::kDouble).contiguous().reshape({-1});
    if (cauchy_tensor.numel() <= 0 || ddsdde_tensor.numel() <= 0)
    {
        return 111;
    }

    cauchy.resize(static_cast<size_t>(cauchy_tensor.numel()));
    ddsdde.resize(static_cast<size_t>(ddsdde_tensor.numel()));
    std::memcpy(cauchy.data(), cauchy_tensor.data_ptr<double>(), cauchy.size() * sizeof(double));
    std::memcpy(ddsdde.data(), ddsdde_tensor.data_ptr<double>(), ddsdde.size() * sizeof(double));
    return 0;
}

static int decode_vumat_results(const torch::jit::IValue &results,
                                int nblock,
                                int nstress,
                                std::vector<double> &energy,
                                std::vector<double> &stress)
{
    if (!results.isTuple())
    {
        return 111;
    }

    auto result_tuple = results.toTuple();
    const auto &elements = result_tuple->elements();
    if (elements.size() < 2)
    {
        return 111;
    }

    const auto &energy_ivalue = elements[0];
    if (energy_ivalue.isTensor())
    {
        auto e = energy_ivalue.toTensor().to(torch::kCPU).to(torch::kDouble).contiguous().reshape({-1});
        if (e.numel() != nblock)
        {
            return 111;
        }
        std::memcpy(energy.data(), e.data_ptr<double>(), static_cast<size_t>(nblock) * sizeof(double));
    }
    else if (energy_ivalue.isDouble() && nblock == 1)
    {
        energy[0] = energy_ivalue.toDouble();
    }
    else
    {
        return 111;
    }

    if (!elements[1].isTensor())
    {
        return 111;
    }

    auto s = elements[1].toTensor().to(torch::kCPU).to(torch::kDouble).contiguous().reshape({nblock, nstress});
    if (s.numel() != static_cast<int64_t>(nblock) * static_cast<int64_t>(nstress))
    {
        return 111;
    }

    auto s_fortran = s.t().contiguous();
    std::memcpy(stress.data(), s_fortran.data_ptr<double>(), stress.size() * sizeof(double));
    return 0;
}

static int handle_umat_request(const std::vector<char> &req, std::vector<char> &resp)
{
    size_t off = 0;
    uint32_t module_len = 0;
    int32_t n_mat_par = 0;

    if (!abqnn::ipc::read_scalar(req, off, module_len)) return 123;
    if (off + module_len > req.size()) return 123;

    std::string module_name(req.data() + off, req.data() + off + module_len);
    off += module_len;

    if (!abqnn::ipc::read_scalar(req, off, n_mat_par)) return 123;
    if (n_mat_par < 0) return 123;
    if (off + 9 * sizeof(double) + static_cast<size_t>(n_mat_par) * sizeof(double) != req.size()) return 123;

    const double *F = reinterpret_cast<const double *>(req.data() + off);
    off += 9 * sizeof(double);
    const double *mat_par = n_mat_par > 0 ? reinterpret_cast<const double *>(req.data() + off) : nullptr;

    torch::jit::Module *mod_ptr = nullptr;
    int mod_load_err = try_load_module(module_name.c_str(), RequestKind::UMAT, mod_ptr);

    int32_t status = mod_load_err;
    double psi = 0.0;
    std::vector<double> cauchy;
    std::vector<double> ddsdde;

    if (status == 0)
    {
        try
        {
            torch::Tensor F_tensor = torch::from_blob((void *)F, {3, 3}, torch::kDouble).t().contiguous();
            torch::Tensor mat_par_tensor = (mat_par && n_mat_par > 0)
                ? torch::from_blob((void *)mat_par, {n_mat_par}, torch::kDouble).contiguous()
                : torch::empty({0}, torch::kDouble);

            auto inference_device = get_inference_device(RequestKind::UMAT);
            F_tensor = F_tensor.to(inference_device);
            mat_par_tensor = mat_par_tensor.to(inference_device);

            auto results = mod_ptr->forward({F_tensor, mat_par_tensor});
            status = decode_umat_results(results, psi, cauchy, ddsdde);
        }
        catch (const std::exception &e)
        {
#ifdef ENABLE_DEBUG_OUTPUT
            std::fprintf(stderr, "server: UMAT inference error: %s\n", e.what());
#endif
            status = 105;
        }
    }

    abqnn::ipc::append_scalar(resp, status);
    if (status == 0)
    {
        int32_t cauchy_n = static_cast<int32_t>(cauchy.size());
        int32_t ddsdde_n = static_cast<int32_t>(ddsdde.size());
        abqnn::ipc::append_scalar(resp, psi);
        abqnn::ipc::append_scalar(resp, cauchy_n);
        abqnn::ipc::append_scalar(resp, ddsdde_n);
        abqnn::ipc::append_bytes(resp, cauchy.data(), cauchy.size() * sizeof(double));
        abqnn::ipc::append_bytes(resp, ddsdde.data(), ddsdde.size() * sizeof(double));
    }

    return 0;
}

static int handle_vumat_request(const std::vector<char> &req, std::vector<char> &resp)
{
    size_t off = 0;
    uint32_t module_len = 0;
    int32_t nblock = 0, ndir = 0, nshr = 0, n_mat_par = 0;

    if (!abqnn::ipc::read_scalar(req, off, module_len)) return 123;
    if (off + module_len > req.size()) return 123;

    std::string module_name(req.data() + off, req.data() + off + module_len);
    off += module_len;

    if (!abqnn::ipc::read_scalar(req, off, nblock) || !abqnn::ipc::read_scalar(req, off, ndir) || !abqnn::ipc::read_scalar(req, off, nshr) || !abqnn::ipc::read_scalar(req, off, n_mat_par)) return 123;
    if (nblock <= 0 || n_mat_par < 0) return 123;

    const size_t ndefgrad = static_cast<size_t>(nblock) * static_cast<size_t>(ndir + 2 * nshr);
    if (off + ndefgrad * sizeof(double) + static_cast<size_t>(n_mat_par) * sizeof(double) != req.size()) return 123;

    const double *defgradF = reinterpret_cast<const double *>(req.data() + off);
    off += ndefgrad * sizeof(double);
    const double *mat_par = n_mat_par > 0 ? reinterpret_cast<const double *>(req.data() + off) : nullptr;

    torch::jit::Module *mod_ptr = nullptr;
    int mod_load_err = try_load_module(module_name.c_str(), RequestKind::VUMAT, mod_ptr);

    int32_t status = mod_load_err;
    const int nstress = ndir + nshr;
    std::vector<double> energy(static_cast<size_t>(nblock), 0.0);
    std::vector<double> stress(static_cast<size_t>(nblock) * static_cast<size_t>(nstress), 0.0);

    if (status == 0)
    {
        try
        {
            torch::Tensor F_batch_tensor;
            status = build_defgrad_batch_tensor(defgradF, nblock, ndir, nshr, F_batch_tensor);

            if (status == 0)
            {
                torch::Tensor mat_par_tensor = (mat_par && n_mat_par > 0)
                    ? torch::from_blob((void *)mat_par, {n_mat_par}, torch::kDouble).contiguous()
                    : torch::empty({0}, torch::kDouble);

                auto inference_device = get_inference_device(RequestKind::VUMAT);
                F_batch_tensor = F_batch_tensor.to(inference_device);
                mat_par_tensor = mat_par_tensor.to(inference_device);

                auto results = mod_ptr->forward({F_batch_tensor, mat_par_tensor});
                status = decode_vumat_results(results, nblock, nstress, energy, stress);
            }
        }
        catch (const std::exception &e)
        {
#ifdef ENABLE_DEBUG_OUTPUT
            std::fprintf(stderr, "server: VUMAT inference error: %s\n", e.what());
#endif
            status = 105;
        }
    }

    abqnn::ipc::append_scalar(resp, status);
    if (status == 0)
    {
        abqnn::ipc::append_scalar(resp, nblock);
        abqnn::ipc::append_scalar(resp, ndir);
        abqnn::ipc::append_scalar(resp, nshr);
        abqnn::ipc::append_bytes(resp, energy.data(), energy.size() * sizeof(double));
        abqnn::ipc::append_bytes(resp, stress.data(), stress.size() * sizeof(double));
    }

    return 0;
}

static int handle_client(HANDLE pipe)
{
    AbqnnIpcHeader req_hdr{};
    if (!abqnn::ipc::read_all(pipe, &req_hdr, sizeof(req_hdr)))
    {
        return 1;
    }

    if (req_hdr.magic != ABQNN_IPC_MAGIC ||
        req_hdr.version != ABQNN_IPC_VERSION ||
        req_hdr.payload_size > ABQNN_IPC_MAX_PAYLOAD)
    {
        return 1;
    }

    std::vector<char> req(req_hdr.payload_size);
    if (req_hdr.payload_size > 0 && !abqnn::ipc::read_all(pipe, req.data(), req.size()))
    {
        return 1;
    }

    std::vector<char> resp;
    uint32_t resp_type = 0;

    switch (req_hdr.message_type)
    {
    case ABQNN_MSG_UMAT_REQ:
        resp_type = ABQNN_MSG_UMAT_RESP;
        handle_umat_request(req, resp);
        break;
    case ABQNN_MSG_VUMAT_REQ:
        resp_type = ABQNN_MSG_VUMAT_RESP;
        handle_vumat_request(req, resp);
        break;
    default:
        return 1;
    }

    AbqnnIpcHeader resp_hdr{};
    resp_hdr.magic = ABQNN_IPC_MAGIC;
    resp_hdr.version = ABQNN_IPC_VERSION;
    resp_hdr.message_type = resp_type;
    resp_hdr.payload_size = static_cast<uint32_t>(resp.size());

    if (!abqnn::ipc::write_all(pipe, &resp_hdr, sizeof(resp_hdr)) ||
        (!resp.empty() && !abqnn::ipc::write_all(pipe, resp.data(), resp.size())))
    {
        return 1;
    }

    return 0;
}

int main()
{
#ifdef ENABLE_DEBUG_OUTPUT
    std::filesystem::create_directories(ABQNN_LOG_PATH);
    auto log_file = (std::filesystem::path(ABQNN_LOG_PATH) / "ipc_server_err.txt").string();
    std::freopen(log_file.c_str(), "a", stderr);
    std::fprintf(stderr, "ABQnn IPC server starting...\n");
    std::fprintf(stderr, "ABQnn UMAT device: %s\n", ABQNN_UMAT_TORCH_DEVICE);
    std::fprintf(stderr, "ABQnn VUMAT device: %s\n", ABQNN_VUMAT_TORCH_DEVICE);
#endif

    int device_err = validate_inference_devices();
    if (device_err != 0)
    {
        return device_err;
    }

    auto serve_client = [](HANDLE pipe)
    {
        handle_client(pipe);
        FlushFileBuffers(pipe);
        DisconnectNamedPipe(pipe);
        CloseHandle(pipe);
    };

    while (true)
    {
        HANDLE pipe = CreateNamedPipeA(
            ABQNN_DEFAULT_PIPE_NAME,
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            ABQNN_IPC_MAX_PAYLOAD,
            ABQNN_IPC_MAX_PAYLOAD,
            0,
            NULL);

        if (pipe == INVALID_HANDLE_VALUE)
        {
            return 2;
        }

        BOOL connected = ConnectNamedPipe(pipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        if (connected)
        {
            std::thread(serve_client, pipe).detach();
        }
        else
        {
            CloseHandle(pipe);
        }
    }

    return 0;
}
