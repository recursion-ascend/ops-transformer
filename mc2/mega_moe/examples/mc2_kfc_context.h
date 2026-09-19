/**
 * \file mc2_kfc_context.h
 * \brief 为 aclnn MegaMoE example 构建 KFC 通信 context（Ascend910B / A2）。
 *
 * MegaMoE 的 `context` 输入不是简单的 {epRankId}，而是一个完整的 Mc2MoeContext 结构：
 * kernel 会把它 reinterpret 成 Mc2MoeContext*，靠 epHcclBuffer_[rankId] 拿到各卡 peer
 * memory 窗口基址来做 Dispatch/Combine。本头文件用 dlsym 从 libhccl.so 拿到 KFC 相关
 * 内部接口，按 torch_extension/csrc/comm_context.cpp 的 KfcContextBuilder 逻辑构建该结构。
 *
 * 仅用于 910B(A2) 单机 EP 场景的 example / msprof 采集；arch35(950) 走 channel 路径，不适用。
 */
#ifndef MC2_KFC_CONTEXT_H_
#define MC2_KFC_CONTEXT_H_

#include <dlfcn.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include "hccl/hccl.h"

namespace mc2_example {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint8_t COMM_ENGINE_AIV = 4;

// 与 mc2/common/op_kernel/mc2_moe_context.h 中的 Mc2MoeContext 严格一致
struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE];
};

// 内部 HCCL 接口签名（取自 torch_extension/cann_ops_transformer/common/hccl_common.h）
using _HcclKfcAllocOpArgs = HcclResult (*)(void **);
using _HcclKfcFreeOpArgs = HcclResult (*)(void *);
using _HcclKfcOpArgsSetCommEngine = HcclResult (*)(void *, uint8_t);
using _HcclKfcOpArgsSetAlgConfig = HcclResult (*)(void *, char *);
using _HcclCommGetHandleWithName = HcclResult (*)(const char *, HcclComm *);
using _HcclCreateOpResCtx = HcclResult (*)(HcclComm, uint8_t, void *, void **);
using _HcclGetRankId = HcclResult (*)(HcclComm, uint32_t *);
using _HcclGetRankSize = HcclResult (*)(HcclComm, uint32_t *);
using _HcclGetHcclBuffer = HcclResult (*)(HcclComm, void **, uint64_t *);
using _HcclGetRemoteIpcHcclBuf = HcclResult (*)(HcclComm, uint64_t, void **, uint64_t *);

struct HcclKfcFuncs {
    _HcclKfcAllocOpArgs allocOpArgs = nullptr;
    _HcclKfcFreeOpArgs freeOpArgs = nullptr;
    _HcclKfcOpArgsSetCommEngine setCommEngine = nullptr;
    _HcclKfcOpArgsSetAlgConfig setAlgConfig = nullptr;
    _HcclCommGetHandleWithName getHandleWithName = nullptr;
    _HcclCreateOpResCtx createOpResCtx = nullptr;
    _HcclGetRankId getRankId = nullptr;
    _HcclGetRankSize getRankSize = nullptr;
    _HcclGetHcclBuffer getHcclBuffer = nullptr;
    _HcclGetRemoteIpcHcclBuf getRemoteIpcBuf = nullptr;
    bool ok = false;
};

template <typename T>
static bool LoadOne(void *lib, const char *name, T &fn)
{
    fn = reinterpret_cast<T>(dlsym(lib, name));
    if (fn == nullptr) {
        printf("[ERROR] dlsym %s from libhccl.so failed: %s\n", name, dlerror());
        return false;
    }
    return true;
}

inline HcclKfcFuncs LoadHcclKfcFuncs()
{
    HcclKfcFuncs f;
    void *lib = dlopen("libhccl.so", RTLD_LAZY | RTLD_GLOBAL);
    if (lib == nullptr) {
        printf("[ERROR] dlopen libhccl.so failed: %s\n", dlerror());
        return f;
    }
    bool ok = true;
    ok &= LoadOne(lib, "HcclKfcAllocOpArgs", f.allocOpArgs);
    ok &= LoadOne(lib, "HcclKfcFreeOpArgs", f.freeOpArgs);
    ok &= LoadOne(lib, "HcclKfcOpArgsSetCommEngine", f.setCommEngine);
    ok &= LoadOne(lib, "HcclKfcOpArgsSetAlgConfig", f.setAlgConfig);
    ok &= LoadOne(lib, "HcclCreateOpResCtx", f.createOpResCtx);
    ok &= LoadOne(lib, "HcclGetRankId", f.getRankId);
    ok &= LoadOne(lib, "HcclGetRankSize", f.getRankSize);
    ok &= LoadOne(lib, "HcclGetHcclBuffer", f.getHcclBuffer);
    ok &= LoadOne(lib, "HcclGetRemoteIpcHcclBuf", f.getRemoteIpcBuf);
    f.ok = ok;
    return f;
}

// 构建 KFC context，输出为 int32 数组（正好覆盖 sizeof(Mc2MoeContext)），可直接作为 context 输入的 host data。
// 直接传入本 rank 已持有的 HcclComm（单进程多线程场景不能按名字查，否则两线程歧义 → 通信死锁）。
// worldSize 为 EP 通信域大小。成功返回 0。
inline int BuildKfcContextBytes(HcclComm commHandle, uint32_t worldSize, std::vector<int32_t> &outInt32,
                                uint64_t *localPeerBase = nullptr)
{
    HcclKfcFuncs f = LoadHcclKfcFuncs();
    if (!f.ok) {
        return -1;
    }

    Mc2MoeContext ctx;
    std::memset(&ctx, 0, sizeof(ctx));

    // 1. 分配并配置 op args（AIV 引擎 + alltoall 算法）
    void *opArgs = nullptr;
    HcclResult ret = f.allocOpArgs(&opArgs);
    if (ret != HCCL_SUCCESS) { printf("[ERROR] HcclKfcAllocOpArgs ret=%d\n", ret); return -1; }
    ret = f.setCommEngine(opArgs, COMM_ENGINE_AIV);
    if (ret != HCCL_SUCCESS) { printf("[ERROR] HcclKfcOpArgsSetCommEngine ret=%d\n", ret); return -1; }
    char algConfig[] = "AlltoAll=level0:fullmesh;level1:pairwise";
    ret = f.setAlgConfig(opArgs, algConfig);
    if (ret != HCCL_SUCCESS) { printf("[ERROR] HcclKfcOpArgsSetAlgConfig ret=%d\n", ret); return -1; }

    // 2. 用本 rank 的 comm handle 创建 op 资源 context（kfcContextAddr）
    void *opsResCtx = nullptr;
    const uint8_t opType = 8;  // 8: AllToAll
    ret = f.createOpResCtx(commHandle, opType, opArgs, &opsResCtx);
    if (ret != HCCL_SUCCESS) { printf("[ERROR] HcclCreateOpResCtx ret=%d\n", ret); return -1; }
    ctx.kfcContextAddr = reinterpret_cast<uint64_t>(opsResCtx);
    f.freeOpArgs(opArgs);

    // 3. 收集各 rank 的 peer memory 窗口基址
    uint32_t rankId = 0;
    ret = f.getRankId(commHandle, &rankId);
    if (ret != HCCL_SUCCESS) { printf("[ERROR] HcclGetRankId ret=%d\n", ret); return -1; }
    ctx.epRankId = rankId;

    for (uint64_t r = 0; r < worldSize; r++) {
        void *addr = nullptr;
        uint64_t sz = 0;
        if (r == rankId) {
            ret = f.getHcclBuffer(commHandle, &addr, &sz);
        } else {
            ret = f.getRemoteIpcBuf(commHandle, r, &addr, &sz);
        }
        if (ret != HCCL_SUCCESS) { printf("[ERROR] get peer buffer rank=%lu ret=%d\n", r, ret); return -1; }
        ctx.epHcclBuffer_[r] = reinterpret_cast<uint64_t>(addr);
    }

    // 4. 输出本 rank 的 peermem 窗口基址（dump 区就在其头部，供 host 侧读回 stage 时间戳）
    if (localPeerBase != nullptr) {
        *localPeerBase = ctx.epHcclBuffer_[rankId];
    }

    // 5. 打包为 int32 数组
    const size_t nInt32 = sizeof(Mc2MoeContext) / sizeof(int32_t);
    outInt32.resize(nInt32);
    std::memcpy(outInt32.data(), &ctx, sizeof(Mc2MoeContext));
    return 0;
}

}  // namespace mc2_example

#endif  // MC2_KFC_CONTEXT_H_
