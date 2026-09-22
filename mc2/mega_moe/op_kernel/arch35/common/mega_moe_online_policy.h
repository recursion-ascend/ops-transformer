/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root.
 */
#ifndef MEGA_MOE_ONLINE_POLICY_H
#define MEGA_MOE_ONLINE_POLICY_H

#include <stdint.h>

// Experimental branch: opt in and rebuild BOTH host and device components.
#ifndef MEGAMOE_ONLINE_SCHEDULER
#define MEGAMOE_ONLINE_SCHEDULER 0
#endif
#ifndef MEGAMOE_LOAD_AWARE
#define MEGAMOE_LOAD_AWARE MEGAMOE_ONLINE_SCHEDULER
#endif
#ifndef MEGAMOE_READY_AWARE
#define MEGAMOE_READY_AWARE MEGAMOE_ONLINE_SCHEDULER
#endif
#ifndef MEGAMOE_CREDIT_AWARE
#define MEGAMOE_CREDIT_AWARE MEGAMOE_ONLINE_SCHEDULER
#endif
#ifndef MEGAMOE_GMM1_SWIZZLE
#define MEGAMOE_GMM1_SWIZZLE MEGAMOE_ONLINE_SCHEDULER
#endif
#ifndef MEGAMOE_AIV1_ARBITRATION
#define MEGAMOE_AIV1_ARBITRATION MEGAMOE_ONLINE_SCHEDULER
#endif

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define MEGAMOE_POLICY_DEVICE __aicore__
#else
#define MEGAMOE_POLICY_DEVICE
#endif

namespace MegaMoeImpl {
namespace OnlinePolicy {
constexpr uint32_t WINDOW = 8U;
// These are experimental software throttle settings, NOT hardware limits.
constexpr uint32_t CREDIT_LIMIT = 12U;
constexpr uint32_t COMBINE_WATERMARK = 8U;

struct LoadSummary {
    uint64_t groups = 0;
    uint32_t maxGroups = 0;
    uint32_t activeExperts = 0;
    MEGAMOE_POLICY_DEVICE inline void Add(uint32_t rows)
    {
        if (rows == 0U) { return; }
        uint32_t g = rows / 256U + (rows % 256U != 0U);
        groups += g;
        maxGroups = maxGroups > g ? maxGroups : g;
        ++activeExperts;
    }
};

MEGAMOE_POLICY_DEVICE inline uint32_t SelectWave(uint32_t base, const LoadSummary &s)
{
    base = base == 0U ? 1U : base;
    if (s.groups < static_cast<uint64_t>(base) * 4U || base == 1U) { return base; }
    // A hypothesis to benchmark, not a universal skew -> smaller-wave rule.
    bool skew = s.activeExperts > 1U &&
                static_cast<uint64_t>(s.maxGroups) * s.activeExperts >= s.groups * 4U;
    if (skew && s.maxGroups >= static_cast<uint64_t>(base) * 2U) { return base / 2U; }
    // Never enlarge beyond the host's validated wave capacity in this version.
    return base;
}

// One descriptor covers a contiguous run of expert-local 256-row groups.
// No token/weight relocation and no cross-expert combining of tail groups.
struct WindowDesc {
    uint32_t offset = 0;
    uint32_t count = 0;
    MEGAMOE_POLICY_DEVICE inline uint32_t Encode() const { return (offset << 4U) | count; }
};

MEGAMOE_POLICY_DEVICE inline WindowDesc Decode(uint32_t v) { return {v >> 4U, v & 15U}; }

MEGAMOE_POLICY_DEVICE inline WindowDesc SelectReady(uint32_t ready, uint32_t done,
                                                  uint32_t remaining, uint32_t maxRun)
{
    uint32_t bound = remaining < WINDOW ? remaining : WINDOW;
    maxRun = maxRun == 0U ? 1U : maxRun;
    WindowDesc d{};
    // Oldest ready run first. A blocked frontier admits at most WINDOW-1
    // bypass groups, after which it must be serviced before new work enters.
    for (uint32_t i = 0; i < bound; ++i) {
        if (((ready & ~done) & (1U << i)) == 0U) { continue; }
        d.offset = i;
        while (i + d.count < bound && d.count < maxRun &&
               ((ready & ~done) & (1U << (i + d.count))) != 0U) { ++d.count; }
        break;
    }
    return d;
}

struct WindowCursor {
    uint32_t frontier = 0;
    uint32_t done = 0;
    uint32_t decision = 0;
    MEGAMOE_POLICY_DEVICE inline void Retire(WindowDesc d)
    {
        done |= ((1U << d.count) - 1U) << d.offset;
        while ((done & 1U) != 0U) { ++frontier; done >>= 1U; }
        ++decision;
    }
};

MEGAMOE_POLICY_DEVICE inline uint32_t ReadyRunLimit(uint32_t cores, uint32_t nTiles)
{
    if (nTiles == 0U) { return 1U; }
    uint32_t fill = cores / nTiles + (cores % nTiles != 0U);
    fill = fill < 3U ? 3U : fill;
    return fill < WINDOW ? fill : WINDOW;
}

MEGAMOE_POLICY_DEVICE inline bool PreferCombine(int32_t produced, int32_t consumed)
{
    return static_cast<int64_t>(produced) - consumed >= COMBINE_WATERMARK;
}

MEGAMOE_POLICY_DEVICE inline uint32_t SelectSwizzle(uint32_t cores, uint32_t nTiles, uint32_t groups)
{
    if (nTiles == 0U || groups == 0U) { return 1U; }
    uint32_t s = cores / nTiles + (cores % nTiles != 0U);
    s = s < 1U ? 1U : s;
    s = s > 4U ? 4U : s;
    return s > groups ? groups : s;
}
} // namespace OnlinePolicy
} // namespace MegaMoeImpl

#undef MEGAMOE_POLICY_DEVICE
#endif
