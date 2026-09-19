/**
 * \file mc2_stage_profile.h
 * \brief 从 MegaMoE peermem 窗口头部的 ExceptionDump 区读回各 stage 的时钟周期时间戳，
 *        打印 stage 级流水耗时。依赖 kernel 内置的 UpdateStage()（每 stage 记 GetSystemCycle）。
 *        纯 host 侧，不改算子。布局与 mc2/common/op_kernel/mc2_exception_dump.h 严格一致。
 */
#ifndef MC2_STAGE_PROFILE_H_
#define MC2_STAGE_PROFILE_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "acl/acl.h"

namespace mc2_example {

// 与 mc2_exception_dump.h 常量一致
constexpr int32_t DUMP_HEADER_MAGIC = 0x5A5A5A5A;
constexpr size_t DUMP_HEADER_REGION_SIZE = 512;
constexpr size_t DUMP_TILING_REGION_SIZE = 2048;

// 与 mc2_exception_dump.h::Header 一致（只需前若干字段）
struct DumpHeader {
    int32_t headerMagic;
    int32_t majorVersion;
    int32_t minorVersion;
    int32_t patchVersion;
    int32_t opType;
    int32_t stageNum;
    uint32_t tilingDataSize;
    uint32_t numBlocks;
    uint64_t dumpCount;
    // execTimes[...] 之后忽略
};

// arch22 Stage 枚举名（见 mega_moe_exception_dump_policy.h）
static const char *kStageNames[] = {
    "INIT",                        // 0
    "APPLY_XACTIVE_MASK",          // 1
    "MOE_INIT_ROUTING",            // 2
    "ALLGATHER_TOKEN_PER_EXPERT",  // 3
    "CUMSUM_TOKEN_PER_EXPERT",     // 4
    "DISPATCH",                    // 5
    "SWIGLU",                      // 6
    "COMBINE",                     // 7
    "RESET_TOKEN_PER_EXPERT",      // 8
    "CROSS_RANK_SYNC",             // 9
    "UNPERMUTE",                   // 10
};
constexpr int kStageNameCount = sizeof(kStageNames) / sizeof(kStageNames[0]);

// peerBaseDevice: 本 rank 的 peermem 窗口基址（device 地址）；rankId 仅用于打印。
// blockToShow: 要展示的 AIV block 序号（默认 0）。返回 0 成功。
inline int ParseAndPrintStageProfile(void *peerBaseDevice, uint32_t rankId, uint32_t blockToShow = 0)
{
    if (peerBaseDevice == nullptr) {
        printf("[PROF] rank %u: peerBaseDevice is null, skip.\n", rankId);
        return -1;
    }

    // 1. 读回 Header
    DumpHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    aclError ret = aclrtMemcpy(&hdr, sizeof(hdr), peerBaseDevice, sizeof(hdr), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        printf("[PROF] rank %u: read header failed, ret=%d\n", rankId, ret);
        return -1;
    }
    if (hdr.headerMagic != DUMP_HEADER_MAGIC) {
        printf("[PROF] rank %u: dump not enabled (magic=0x%x). kernel 未写 dump，检查是否真的执行。\n",
               rankId, hdr.headerMagic);
        return -1;
    }
    int32_t stageNum = hdr.stageNum;
    if (stageNum <= 0 || stageNum > kStageNameCount) {
        printf("[PROF] rank %u: unexpected stageNum=%d\n", rankId, stageNum);
        return -1;
    }

    // 2. BlockStage 布局: currentStage(4B)+pad(4B)+stageCycle[stageNum](8B each) = 8 + stageNum*8
    const size_t blockStageSize = 8 + static_cast<size_t>(stageNum) * 8;
    const size_t blockStageBase = DUMP_HEADER_REGION_SIZE + DUMP_TILING_REGION_SIZE;
    if (blockToShow >= hdr.numBlocks) {
        blockToShow = 0;
    }

    std::vector<uint8_t> buf(blockStageSize);
    void *blockAddr = static_cast<uint8_t *>(peerBaseDevice) + blockStageBase +
                      static_cast<size_t>(blockToShow) * blockStageSize;
    ret = aclrtMemcpy(buf.data(), blockStageSize, blockAddr, blockStageSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        printf("[PROF] rank %u: read blockStage failed, ret=%d\n", rankId, ret);
        return -1;
    }

    uint32_t currentStage = 0;
    std::memcpy(&currentStage, buf.data(), sizeof(uint32_t));
    const uint64_t *stageCycle = reinterpret_cast<const uint64_t *>(buf.data() + 8);

    // 3. 打印：各 stage 起始 cycle + 到下一个"有效"stage 的差值（= 该 stage 耗时）
    printf("\n========== MegaMoE stage profile (rank %u, AIV block %u, numBlocks=%u) ==========\n",
           rankId, blockToShow, hdr.numBlocks);
    printf("%-28s %16s %16s\n", "STAGE", "start_cycle", "delta_cycle");

    // 找下一个非零 stage 作为结束点计算 delta
    uint64_t total = 0;
    for (int s = 0; s < stageNum; s++) {
        uint64_t cur = stageCycle[s];
        if (cur == 0) {
            printf("%-28s %16s %16s   (未进入)\n", kStageNames[s], "-", "-");
            continue;
        }
        uint64_t next = 0;
        for (int t = s + 1; t < stageNum; t++) {
            if (stageCycle[t] != 0) { next = stageCycle[t]; break; }
        }
        if (next != 0) {
            uint64_t d = next - cur;
            total += d;
            printf("%-28s %16llu %16llu\n", kStageNames[s],
                   (unsigned long long)cur, (unsigned long long)d);
        } else {
            printf("%-28s %16llu %16s   (末阶段/无后继)\n", kStageNames[s], (unsigned long long)cur, "-");
        }
    }
    printf("%-28s %16s %16llu\n", "[sum of deltas]", "", (unsigned long long)total);
    printf("说明: cycle 为 GetSystemCycle() 读数; delta = 下一有效 stage 起点 - 本 stage 起点 = 该 stage 耗时(周期).\n");
    printf("      各 AIV block 处理不同专家, 耗时可能不同; 换其它 block 传 blockToShow.\n");
    printf("====================================================================================\n\n");
    return 0;
}

}  // namespace mc2_example

#endif  // MC2_STAGE_PROFILE_H_
