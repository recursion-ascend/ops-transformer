/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the repository for the full text of the License.
 */

/*!
 * \file test_aclnn_mega_moe.cpp
 * \brief MegaMoE 算子 example，用于 msprof 性能采集
 */

#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <random>
#include <limits>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnn_mega_moe.h"
#include "mc2_kfc_context.h"
#include "mc2_stage_profile.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    HcclComm hcclEpComm;
    aclrtStream stream;
    aclrtContext context;
};

const uint32_t EP_WORLD_SIZE = 2;
const uint32_t DEV_NUM = EP_WORLD_SIZE;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostDataList, const std::vector<int64_t> &shape,
                        std::vector<void*> &deviceAddrs, aclDataType dataType, aclTensorList **tensorList)
{
    std::vector<aclTensor*> tensors(hostDataList.size());
    deviceAddrs.resize(hostDataList.size());
    for (size_t i = 0; i < hostDataList.size(); i++) {
        auto ret = CreateAclTensor(hostDataList[i], shape, &deviceAddrs[i], dataType, &tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensorList = aclCreateTensorList(tensors.data(), tensors.size());
    return 0;
}

void DestroyTensor(aclTensor *tensor)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
}

void DestroyTensorList(aclTensorList *tensorList)
{
    if (tensorList != nullptr) {
        aclDestroyTensorList(tensorList);
    }
}

void FreeDeviceAddr(void *deviceAddr)
{
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int launchOneThreadMegaMoe(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, stream = %p, context = %p\n",
              args.rankId, hcomEpName, args.stream, args.context);

    // 测试参数设置
    int64_t BS = 128;           // batch size (token数)
    int64_t H = 4096;           // hidden dimension
    int64_t K = 8;              // topK
    int64_t moeExpertNum = 16;  // 总专家数
    int64_t localExpertNum = moeExpertNum / EP_WORLD_SIZE;  // 每卡专家数
    int64_t intermediateHidden = 1024;  // FFN中间维度
    int64_t cclBufferSize = 64 * 1024 * 1024;  // CCL缓冲区大小 (需≥54MB, 见 CheckHcclBuffSize)

    // 构造 context: 必须是完整的 Mc2MoeContext (含各 rank 的 peer memory 窗口基址)，
    // 由 HCCL KFC 接口构建，而不是简单的 {epRankId}。见 mc2_kfc_context.h。
    std::vector<int32_t> contextHostData;
    uint64_t localPeerBase = 0;
    ret = mc2_example::BuildKfcContextBytes(args.hcclEpComm, EP_WORLD_SIZE, contextHostData, &localPeerBase);
    CHECK_RET(ret == 0, LOG_PRINT("[ERROR] BuildKfcContextBytes failed.\n"); return -1);
    std::vector<int64_t> contextShape = {static_cast<int64_t>(contextHostData.size())};

    // 构造输入 x: [BS, H]
    std::vector<op::fp16_t> xHostData(BS * H, 1);
    std::vector<int64_t> xShape = {BS, H};

    // 构造 topkIds: [BS, K] - 每个token选择K个专家
    std::vector<int32_t> topkIdsHostData;
    for (int64_t i = 0; i < BS; i++) {
        for (int64_t k = 0; k < K; k++) {
            topkIdsHostData.push_back((i + k) % moeExpertNum);
        }
    }
    std::vector<int64_t> topkIdsShape = {BS, K};

    // 构造 topkWeights: [BS, K]
    std::vector<float> topkWeightsHostData(BS * K, 1.0f / K);
    std::vector<int64_t> topkWeightsShape = {BS, K};

    // 构造 weight1: [localExpertNum] 个 [H, 2*intermediateHidden] 的矩阵
    std::vector<std::vector<op::fp16_t>> weight1HostDataList(localExpertNum);
    std::vector<int64_t> weight1Shape = {H, 2 * intermediateHidden};
    for (int64_t e = 0; e < localExpertNum; e++) {
        weight1HostDataList[e].resize(H * 2 * intermediateHidden, 0.01);
    }

    // 构造 weight2: [localExpertNum] 个 [intermediateHidden, H] 的矩阵
    std::vector<std::vector<op::fp16_t>> weight2HostDataList(localExpertNum);
    std::vector<int64_t> weight2Shape = {intermediateHidden, H};
    for (int64_t e = 0; e < localExpertNum; e++) {
        weight2HostDataList[e].resize(intermediateHidden * H, 0.01);
    }

    // 构造输出 y: [BS, H]
    std::vector<op::fp16_t> yHostData(BS * H, 0);
    std::vector<int64_t> yShape = {BS, H};

    // 构造 expertTokenNumsOut: [localExpertNum]
    std::vector<int32_t> expertTokenNumsHostData(localExpertNum, 0);
    std::vector<int64_t> expertTokenNumsShape = {localExpertNum};

    // 创建 device 侧 tensor
    void *contextDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    void *topkIdsDeviceAddr = nullptr;
    void *topkWeightsDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;

    aclTensor *context = nullptr;
    aclTensor *x = nullptr;
    aclTensor *topkIds = nullptr;
    aclTensor *topkWeights = nullptr;
    aclTensor *y = nullptr;
    aclTensor *expertTokenNumsOut = nullptr;

    ret = CreateAclTensor(contextHostData, contextShape, &contextDeviceAddr, aclDataType::ACL_INT32, &context);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkIdsHostData, topkIdsShape, &topkIdsDeviceAddr, aclDataType::ACL_INT32, &topkIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkWeightsHostData, topkWeightsShape, &topkWeightsDeviceAddr, aclDataType::ACL_FLOAT, &topkWeights);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr,
                          aclDataType::ACL_INT32, &expertTokenNumsOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 weight1 和 weight2 TensorList
    aclTensorList *weight1 = nullptr;
    aclTensorList *weight2 = nullptr;
    std::vector<void*> weight1DeviceAddrs;
    std::vector<void*> weight2DeviceAddrs;

    ret = CreateAclTensorList(weight1HostDataList, weight1Shape, weight1DeviceAddrs, aclDataType::ACL_BF16, &weight1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList(weight2HostDataList, weight2Shape, weight2DeviceAddrs, aclDataType::ACL_BF16, &weight2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造 activationParams: swiglu 需要非空的 aclFloatArray；传 1 个 clamp 值
    // (FLT_MAX 等价于默认不裁剪，见 tiling 的 DEFAULT_ACTIVATION_CLAMP)
    std::vector<float> activationParamsData = {std::numeric_limits<float>::max()};
    aclFloatArray *activationParams = aclCreateFloatArray(activationParamsData.data(),
                                                          activationParamsData.size());
    CHECK_RET(activationParams != nullptr, return -1);

    // 调用 aclnnMegaMoe
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    // 第一阶段：获取 workspace 大小
    ret = aclnnMegaMoeGetWorkspaceSize(
        context, x, topkIds, topkWeights,
        weight1, weight2,
        nullptr, nullptr,  // weightScales1, weightScales2 (非量化场景不需要)
        nullptr, nullptr,  // bias1, bias2
        nullptr,           // xActiveMask
        nullptr, nullptr,  // sharedWeight1, sharedWeight2
        nullptr, nullptr,  // sharedWeightScales1, sharedWeightScales2
        nullptr, nullptr,  // sharedBias1, sharedBias2
        nullptr,           // maskBuffer
        moeExpertNum, EP_WORLD_SIZE, cclBufferSize,
        0,   // maxRecvTokenNum
        0,   // dispatchQuantMode (非量化)
        27,  // dispatchQuantOutDtype (BF16 场景需为 27 = ge::DT_BF16)
        0,   // combineQuantMode
        "",  // commAlg
        0,   // numMaxTokensPerRank
        "swiglu",  // activation
        activationParams,  // activationParams
        0,   // topoType
        2,   // rankNumPerServer
        0,   // topkWeightsType
        y, expertTokenNumsOut,
        &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMegaMoeGetWorkspaceSize failed. ret = %d\n", ret);
              return ret);

    // 申请 workspace
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d\n", ret); return ret);
    }

    // 第二阶段：执行算子
    ret = aclnnMegaMoe(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMegaMoe failed. ret = %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 30000);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret);
              return ret);

    LOG_PRINT("[INFO] device_%d aclnnMegaMoe execute successfully.\n", args.rankId);

    // 读回 kernel 内置 stage 时间戳（peermem dump 区），打印 stage 级流水耗时
    mc2_example::ParseAndPrintStageProfile(reinterpret_cast<void *>(localPeerBase), args.rankId, 0);

    // 释放资源
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    DestroyTensor(context);
    DestroyTensor(x);
    DestroyTensor(topkIds);
    DestroyTensor(topkWeights);
    DestroyTensor(y);
    DestroyTensor(expertTokenNumsOut);
    DestroyTensorList(weight1);
    DestroyTensorList(weight2);

    FreeDeviceAddr(contextDeviceAddr);
    FreeDeviceAddr(xDeviceAddr);
    FreeDeviceAddr(topkIdsDeviceAddr);
    FreeDeviceAddr(topkWeightsDeviceAddr);
    FreeDeviceAddr(yDeviceAddr);
    FreeDeviceAddr(expertTokenNumsDeviceAddr);
    for (auto addr : weight1DeviceAddrs) FreeDeviceAddr(addr);
    for (auto addr : weight2DeviceAddrs) FreeDeviceAddr(addr);

    HcclCommDestroy(args.hcclEpComm);
    aclrtDestroyStream(args.stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

// 进程间共享 HcclRootInfo：rank0 生成，其余 rank 自旋等待后读取
struct SharedInit {
    HcclRootInfo rootInfo;
    volatile int ready;
};

// 单个 rank 在独立进程内执行（每进程独占一张卡）
int runOneRank(uint32_t rankId, SharedInit *shared)
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %u aclInit failed. ret = %d\n", rankId, ret); return ret);
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %u aclrtSetDevice failed. ret = %d\n", rankId, ret); return ret);

    aclrtContext context = nullptr;
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %u aclrtCreateContext failed. ret = %d\n", rankId, ret); return ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] rank %u aclrtCreateStream failed. ret = %d\n", rankId, ret); return ret);

    // rank0 生成 rootInfo 并置位；其余 rank 等待
    if (rankId == 0) {
        ret = HcclGetRootInfo(&shared->rootInfo);
        CHECK_RET(ret == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclGetRootInfo failed. ret = %d\n", ret); return ret);
        __sync_synchronize();
        shared->ready = 1;
    } else {
        while (shared->ready == 0) {
            usleep(1000);
        }
        __sync_synchronize();
    }

    HcclComm comm = nullptr;
    ret = HcclCommInitRootInfo(EP_WORLD_SIZE, &shared->rootInfo, rankId, &comm);
    CHECK_RET(ret == HCCL_SUCCESS, LOG_PRINT("[ERROR] rank %u HcclCommInitRootInfo failed. ret = %d\n", rankId, ret); return ret);

    Args args;
    args.rankId = rankId;
    args.epRankId = rankId;
    args.hcclEpComm = comm;
    args.stream = stream;
    args.context = context;

    ret = launchOneThreadMegaMoe(args);

    aclFinalize();
    return ret;
}

int main(int argc, char *argv[])
{
    // fork 前建立匿名共享内存，用于跨进程传递 HcclRootInfo
    SharedInit *shared = static_cast<SharedInit *>(
        mmap(nullptr, sizeof(SharedInit), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    CHECK_RET(shared != MAP_FAILED, LOG_PRINT("[ERROR] mmap SharedInit failed.\n"); return -1);
    shared->ready = 0;

    pid_t pids[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        pid_t pid = fork();
        CHECK_RET(pid >= 0, LOG_PRINT("[ERROR] fork failed.\n"); return -1);
        if (pid == 0) {
            int rc = runOneRank(rankId, shared);
            _exit(rc == 0 ? 0 : 1);
        }
        pids[rankId] = pid;
    }

    int failed = 0;
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        int status = 0;
        waitpid(pids[rankId], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            failed = 1;
            LOG_PRINT("[ERROR] rank %u process exited abnormally (status=%d).\n", rankId, status);
        }
    }

    munmap(shared, sizeof(SharedInit));
    if (failed) {
        LOG_PRINT("[ERROR] MegaMoE example failed.\n");
        return 1;
    }
    LOG_PRINT("[INFO] MegaMoE example completed successfully.\n");
    return 0;
}
