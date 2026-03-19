#ifndef ABQNN_IPC_COMMON_H
#define ABQNN_IPC_COMMON_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace abqnn::ipc {

static constexpr int ERR_IPC_CONNECT = 120;
static constexpr int ERR_IPC_WRITE = 121;
static constexpr int ERR_IPC_READ = 122;
static constexpr int ERR_IPC_PROTOCOL = 123;

bool write_all(HANDLE h, const void* data, size_t n);
bool read_all(HANDLE h, void* data, size_t n);

int transact_blocking(const char* pipe_name,
                     uint32_t request_type,
                     const std::vector<char>& request_payload,
                     uint32_t expected_response_type,
                     std::vector<char>& response_payload);

template <typename T>
inline void append_scalar(std::vector<char>& buf, const T& v)
{
    size_t old_size = buf.size();
    buf.resize(old_size + sizeof(T));
    std::memcpy(buf.data() + old_size, &v, sizeof(T));
}

inline void append_bytes(std::vector<char>& buf, const void* data, size_t n)
{
    size_t old_size = buf.size();
    buf.resize(old_size + n);
    std::memcpy(buf.data() + old_size, data, n);
}

template <typename T>
inline bool read_scalar(const std::vector<char>& buf, size_t& offset, T& out)
{
    if (offset + sizeof(T) > buf.size())
    {
        return false;
    }
    std::memcpy(&out, buf.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

} // namespace abqnn::ipc

#endif // ABQNN_IPC_COMMON_H
