#ifndef ABQNN_IPC_PROTOCOL_H
#define ABQNN_IPC_PROTOCOL_H

#include <cstdint>

static constexpr uint32_t ABQNN_IPC_MAGIC = 0x4E4E5141; // 'AQNN'
static constexpr uint32_t ABQNN_IPC_VERSION = 1;
static constexpr uint32_t ABQNN_IPC_MAX_PAYLOAD = 256u * 1024u * 1024u; // 256 MB

static constexpr const char* ABQNN_DEFAULT_PIPE_NAME = "\\\\.\\pipe\\abqnn_inference";

enum AbqnnIpcMessageType : uint32_t {
    ABQNN_MSG_UMAT_REQ = 1,
    ABQNN_MSG_UMAT_RESP = 2,
    ABQNN_MSG_VUMAT_REQ = 3,
    ABQNN_MSG_VUMAT_RESP = 4,
};

#pragma pack(push, 1)
struct AbqnnIpcHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t message_type;
    uint32_t payload_size;
};
#pragma pack(pop)

#endif // ABQNN_IPC_PROTOCOL_H
