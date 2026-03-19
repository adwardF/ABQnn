#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <filesystem>
#include <vector>

#include <ctime>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

#include "umat_auxlib.h"
#include "abqnn_config.h"
#include "abqnn_ipc_protocol.h"
#include "abqnn_ipc_common.h"

static std::once_flag init_flag;
static int initialization_error = 0;
static const char *kPipeName = ABQNN_DEFAULT_PIPE_NAME;

static int initialize_library()
{
#ifdef ENABLE_DEBUG_OUTPUT
    std::filesystem::create_directories(ABQNN_LOG_PATH);
    char log_file_path[MAX_PATH];
    std::snprintf(log_file_path, MAX_PATH, "%s/auxlib_err.txt", ABQNN_LOG_PATH);
    std::freopen(log_file_path, "a", stderr);

    time_t now = time(NULL);
    std::fprintf(stderr, "UMAT_auxlib.cpp: %s", ctime(&now));
    std::fprintf(stderr, "Initializing IPC client.\n");
    std::fprintf(stderr, "Pipe endpoint: %s\n", ABQNN_DEFAULT_PIPE_NAME);
#endif
    return 0;
}

int invoke_pt(const char *module_filename,
              const double *F, const double *mat_par, int n_mat_par,
              double *psi, double *Cauchy, double *DDSDDE)
{
    std::call_once(init_flag, []() {
        initialization_error = initialize_library();
    });
    
    // Check if initialization failed
    if (initialization_error != 0)
    {
        return initialization_error;
    }
    
    if (!module_filename || !F || !psi || !Cauchy || !DDSDDE)
    {
        return 110;
    }

    uint32_t module_len = static_cast<uint32_t>(std::strlen(module_filename));
    int32_t n_mat_par_i32 = static_cast<int32_t>(n_mat_par);

    std::vector<char> req;
    req.reserve(sizeof(module_len) + module_len + sizeof(n_mat_par_i32) + 9 * sizeof(double) +
                static_cast<size_t>(n_mat_par > 0 ? n_mat_par : 0) * sizeof(double));

    abqnn::ipc::append_scalar(req, module_len);
    abqnn::ipc::append_bytes(req, module_filename, module_len);
    abqnn::ipc::append_scalar(req, n_mat_par_i32);
    abqnn::ipc::append_bytes(req, F, 9 * sizeof(double));
    if (mat_par && n_mat_par > 0)
    {
        abqnn::ipc::append_bytes(req, mat_par, static_cast<size_t>(n_mat_par) * sizeof(double));
    }

    std::vector<char> resp;
    int tx_err = abqnn::ipc::transact_blocking(kPipeName, ABQNN_MSG_UMAT_REQ, req, ABQNN_MSG_UMAT_RESP, resp);
    if (tx_err != 0)
    {
        return tx_err;
    }

    size_t off = 0;
    int32_t status = 0;
    if (!abqnn::ipc::read_scalar(resp, off, status))
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }
    if (status != 0)
    {
        return status;
    }

    if (!abqnn::ipc::read_scalar(resp, off, *psi))
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }

    int32_t cauchy_n = 0;
    int32_t ddsdde_n = 0;
    if (!abqnn::ipc::read_scalar(resp, off, cauchy_n) || !abqnn::ipc::read_scalar(resp, off, ddsdde_n))
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }

    if (cauchy_n <= 0 || ddsdde_n <= 0)
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }

    size_t cauchy_bytes = static_cast<size_t>(cauchy_n) * sizeof(double);
    size_t ddsdde_bytes = static_cast<size_t>(ddsdde_n) * sizeof(double);

    if (off + cauchy_bytes + ddsdde_bytes != resp.size())
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }

    std::memcpy(Cauchy, resp.data() + off, cauchy_bytes);
    off += cauchy_bytes;
    std::memcpy(DDSDDE, resp.data() + off, ddsdde_bytes);

    return 0;
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

    if (!module_filename || !defgradF || !enerInternNew || !stressNew || nblock <= 0)
    {
        return 110;
    }

    if (!((ndir == 3 && nshr == 3) || (ndir == 3 && nshr == 1)))
    {
        return 111;
    }

    uint32_t module_len = static_cast<uint32_t>(std::strlen(module_filename));
    int32_t nblock_i32 = static_cast<int32_t>(nblock);
    int32_t ndir_i32 = static_cast<int32_t>(ndir);
    int32_t nshr_i32 = static_cast<int32_t>(nshr);
    int32_t n_mat_par_i32 = static_cast<int32_t>(n_mat_par);

    const size_t ndefgrad = static_cast<size_t>(nblock) * static_cast<size_t>(ndir + 2 * nshr);
    const size_t nstress = static_cast<size_t>(nblock) * static_cast<size_t>(ndir + nshr);

    std::vector<char> req;
    req.reserve(sizeof(module_len) + module_len +
                sizeof(nblock_i32) + sizeof(ndir_i32) + sizeof(nshr_i32) + sizeof(n_mat_par_i32) +
                ndefgrad * sizeof(double) +
                static_cast<size_t>(n_mat_par > 0 ? n_mat_par : 0) * sizeof(double));

    abqnn::ipc::append_scalar(req, module_len);
    abqnn::ipc::append_bytes(req, module_filename, module_len);
    abqnn::ipc::append_scalar(req, nblock_i32);
    abqnn::ipc::append_scalar(req, ndir_i32);
    abqnn::ipc::append_scalar(req, nshr_i32);
    abqnn::ipc::append_scalar(req, n_mat_par_i32);
    abqnn::ipc::append_bytes(req, defgradF, ndefgrad * sizeof(double));
    if (mat_par && n_mat_par > 0)
    {
        abqnn::ipc::append_bytes(req, mat_par, static_cast<size_t>(n_mat_par) * sizeof(double));
    }

    std::vector<char> resp;
    int tx_err = abqnn::ipc::transact_blocking(kPipeName, ABQNN_MSG_VUMAT_REQ, req, ABQNN_MSG_VUMAT_RESP, resp);
    if (tx_err != 0)
    {
        return tx_err;
    }

    size_t off = 0;
    int32_t status = 0;
    if (!abqnn::ipc::read_scalar(resp, off, status))
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }
    if (status != 0)
    {
        return status;
    }

    int32_t resp_nblock = 0;
    int32_t resp_ndir = 0;
    int32_t resp_nshr = 0;
    if (!abqnn::ipc::read_scalar(resp, off, resp_nblock) || !abqnn::ipc::read_scalar(resp, off, resp_ndir) || !abqnn::ipc::read_scalar(resp, off, resp_nshr))
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }

    if (resp_nblock != nblock_i32 || resp_ndir != ndir_i32 || resp_nshr != nshr_i32)
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }

    if (off + static_cast<size_t>(nblock) * sizeof(double) + nstress * sizeof(double) != resp.size())
    {
        return abqnn::ipc::ERR_IPC_PROTOCOL;
    }

    std::memcpy(enerInternNew, resp.data() + off, static_cast<size_t>(nblock) * sizeof(double));
    off += static_cast<size_t>(nblock) * sizeof(double);
    std::memcpy(stressNew, resp.data() + off, nstress * sizeof(double));

    return 0;
}