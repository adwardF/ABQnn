#include "abqnn_ipc_common.h"
#include "abqnn_ipc_protocol.h"

namespace abqnn::ipc {

bool write_all(HANDLE h, const void* data, size_t n)
{
    const char* p = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < n)
    {
        DWORD wrote = 0;
        if (!WriteFile(h, p + sent, static_cast<DWORD>(n - sent), &wrote, NULL) || wrote == 0)
        {
            return false;
        }
        sent += wrote;
    }
    return true;
}

bool read_all(HANDLE h, void* data, size_t n)
{
    char* p = static_cast<char*>(data);
    size_t got = 0;
    while (got < n)
    {
        DWORD read_n = 0;
        if (!ReadFile(h, p + got, static_cast<DWORD>(n - got), &read_n, NULL) || read_n == 0)
        {
            return false;
        }
        got += read_n;
    }
    return true;
}

int transact_blocking(const char* pipe_name,
                     uint32_t request_type,
                     const std::vector<char>& request_payload,
                     uint32_t expected_response_type,
                     std::vector<char>& response_payload)
{
    HANDLE pipe = CreateFileA(
        pipe_name,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        0,
        NULL);

    if (pipe == INVALID_HANDLE_VALUE)
    {
        return ERR_IPC_CONNECT;
    }

    AbqnnIpcHeader req_hdr{};
    req_hdr.magic = ABQNN_IPC_MAGIC;
    req_hdr.version = ABQNN_IPC_VERSION;
    req_hdr.message_type = request_type;
    req_hdr.payload_size = static_cast<uint32_t>(request_payload.size());

    if (!write_all(pipe, &req_hdr, sizeof(req_hdr)) ||
        (!request_payload.empty() && !write_all(pipe, request_payload.data(), request_payload.size())))
    {
        CloseHandle(pipe);
        return ERR_IPC_WRITE;
    }

    AbqnnIpcHeader resp_hdr{};
    if (!read_all(pipe, &resp_hdr, sizeof(resp_hdr)))
    {
        CloseHandle(pipe);
        return ERR_IPC_READ;
    }

    if (resp_hdr.magic != ABQNN_IPC_MAGIC ||
        resp_hdr.version != ABQNN_IPC_VERSION ||
        resp_hdr.message_type != expected_response_type ||
        resp_hdr.payload_size > ABQNN_IPC_MAX_PAYLOAD)
    {
        CloseHandle(pipe);
        return ERR_IPC_PROTOCOL;
    }

    response_payload.resize(resp_hdr.payload_size);
    if (resp_hdr.payload_size > 0 && !read_all(pipe, response_payload.data(), resp_hdr.payload_size))
    {
        CloseHandle(pipe);
        return ERR_IPC_READ;
    }

    CloseHandle(pipe);
    return 0;
}

} // namespace abqnn::ipc
