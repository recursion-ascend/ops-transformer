// CPU-only tests for the actual policy and scheduler geometry headers.
// Does not claim CANN compilation, NPU memory ordering, or numerical accuracy.
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <vector>
#include "../../op_kernel/arch35/common/mega_moe_online_policy.h"
#include "../../op_kernel/arch35/blaze/gemm/block/block_scheduler_swizzle.h"

using namespace MegaMoeImpl::OnlinePolicy;
using Scheduler = Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 0>;

static void CheckLoads()
{
    LoadSummary empty{};
    assert(SelectWave(0, empty) == 1);
    assert(SelectWave(8, empty) == 8);
    LoadSummary tails{};
    for (auto rows : {0U, 1U, 255U, 256U, 257U}) { tails.Add(rows); }
    assert(tails.groups == 5 && tails.maxGroups == 2 && tails.activeExperts == 4);
    LoadSummary balanced{};
    for (int e = 0; e < 16; ++e) { balanced.Add(4096); }
    assert(SelectWave(8, balanced) == 8);
    LoadSummary skew{};
    skew.Add(25600);
    for (int e = 0; e < 15; ++e) { skew.Add(256); }
    assert(SelectWave(8, skew) == 4);
    assert(SelectWave(1, skew) == 1);
    LoadSummary one{};
    one.Add(25600);
    assert(SelectWave(8, one) == 8);
    LoadSummary big{};
    for (int e = 0; e < 1024; ++e) { big.Add(UINT32_MAX); }
    assert(big.groups == uint64_t(1024) * 16777216U);
    assert(SelectSwizzle(28, 16, 8) == 2);
}

static void CheckWindows()
{
    // Exhaust every readiness/completion combination in the bounded window.
    for (uint32_t remaining = 1; remaining <= WINDOW; ++remaining) {
        for (uint32_t ready = 0; ready < 256; ++ready) {
            for (uint32_t done = 0; done < 256; ++done) {
                auto d = SelectReady(ready, done, remaining, 3);
                assert(d.count <= 3);
                if (d.count == 0) {
                    assert(((ready & ~done) & ((1U << remaining) - 1)) == 0);
                    continue;
                }
                assert(d.offset + d.count <= remaining);
                uint32_t selected = ((1U << d.count) - 1) << d.offset;
                assert((selected & done) == 0 && (selected & ready) == selected);
                assert(Decode(d.Encode()).offset == d.offset);
                assert(Decode(d.Encode()).count == d.count);
            }
        }
    }
    WindowCursor c{};
    auto d = SelectReady(254, c.done, 20, 8);
    assert(d.offset == 1 && d.count == 7);
    c.Retire(d);
    assert(c.frontier == 0 && c.done == 254);
    assert(SelectReady(254, c.done, 20, 8).count == 0); // bounded bypass
    c.Retire({0, 1});
    assert(c.frontier == 8 && c.done == 0);
}

static void CheckReplayAndGeometry()
{
    std::mt19937 rng(1733);
    for (unsigned trial = 0; trial < 10000; ++trial) {
        uint32_t groups = 1 + rng() % 80;
        uint32_t tail = 1 + rng() % 256;
        uint32_t rows = (groups - 1) * 256 + tail;
        uint32_t n = 1 + rng() % 2048;
        uint32_t nt = (n + 255) / 256;
        uint32_t cores = 1 + rng() % 32;
        uint32_t start = rng() % cores;
        uint32_t initialStart = start;
        WindowCursor aic{}, aiv{};
        std::set<std::pair<uint32_t, uint32_t>> tiles;
        uint32_t reversal = 0;
        while (aic.frontier < groups) {
            auto d = SelectReady(rng() & 255, aic.done, groups - aic.frontier,
                                 ReadyRunLimit(cores, nt));
            if (d.count == 0) { d = {0, 1}; } // eventual Dispatch of oldest group
            auto published = Decode(d.Encode());
            assert(published.offset == d.offset && published.count == d.count);
            uint32_t first = aic.frontier + d.offset;
            uint32_t offset = first * 256;
            uint32_t m = std::min(rows - offset, d.count * 256);
            uint32_t s = SelectSwizzle(cores, nt, d.count);
            Scheduler scheduler({m, n, 128}, {{256, 256}, offset, 0, (reversal & 1) != 0, s});
            uint32_t count = scheduler.GetTileNum();
            assert(count == d.count * nt);
            for (uint32_t core = 0; core < cores; ++core) {
                uint32_t firstTile = (core < start ? core + cores : core) - start;
                for (uint32_t t = firstTile; t < count; t += cores) {
                    auto coord = scheduler.GetBlockCoord(t);
                    auto shape = scheduler.GetBlockShape(coord);
                    auto mr = std::get<0>(coord);
                    auto nc = std::get<1>(coord);
                    assert(mr >= offset && mr < offset + m && nc >= 0 && nc < n);
                    assert(std::get<0>(shape) == std::min<int64_t>(256, offset + m - mr));
                    assert(std::get<1>(shape) == std::min<int64_t>(256, n - nc));
                    assert(tiles.emplace(mr / 256, nc / 256).second);
                }
            }
            start = (start + count) % cores;
            reversal += (d.count + s - 1) / s;
            aic.Retire(d);
            aiv.Retire(published);
            assert(aic.frontier == aiv.frontier && aic.done == aiv.done);
            assert(aic.decision <= groups);
        }
        assert(tiles.size() == groups * nt);
        assert(start == (initialStart + groups * nt) % cores);
    }
}

static void CheckCreditArithmetic()
{
    std::mt19937 rng(33);
    for (int trial = 0; trial < 1000; ++trial) {
        int produced = 0, consumed = 0;
        for (int t = 0; t < 10000; ++t) {
            bool issue = (rng() & 1) && produced - consumed < int(CREDIT_LIMIT);
            if (issue) { ++produced; }
            else if (consumed < produced) { ++consumed; }
            assert(produced >= consumed && produced - consumed <= int(CREDIT_LIMIT));
            assert(PreferCombine(produced, consumed) == (produced - consumed >= int(COMBINE_WATERMARK)));
        }
    }
}

static void CheckDefaultSwizzle()
{
    using NScheduler = Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>;
    for (int64_t mg = 1; mg < 17; ++mg) {
        for (int64_t ng = 1; ng < 17; ++ng) {
            Scheduler base({mg * 256 - 13, ng * 256 - 7, 128}, {{256, 256}});
            NScheduler ns({mg * 256 - 13, ng * 256 - 7, 128}, {{256, 256}, 256, 512, true, 2});
            std::set<std::pair<int64_t, int64_t>> seen;
            for (int64_t t = 0; t < mg * ng; ++t) {
                int64_t group = t / (3 * ng), local = t % (3 * ng);
                int64_t valid = std::min<int64_t>(3, mg - group * 3);
                int64_t m = group * 3 + local % valid;
                int64_t n = local / valid;
                if (group & 1) { n = ng - n - 1; }
                auto coord = base.GetBlockCoord(t);
                assert(std::get<0>(coord) == m * 256 && std::get<1>(coord) == n * 256);
                auto nc = ns.GetBlockCoord(t);
                auto shape = ns.GetBlockShape(nc);
                assert(std::get<0>(shape) > 0 && std::get<1>(shape) > 0);
                assert(seen.emplace(std::get<0>(nc), std::get<1>(nc)).second);
            }
            assert(seen.size() == static_cast<size_t>(mg * ng));
        }
    }
}

int main()
{
    CheckLoads(); CheckWindows(); CheckReplayAndGeometry(); CheckCreditArithmetic(); CheckDefaultSwizzle();
    std::cout << "PASS: load policy, exhaustive bounded selection, 10000 replay/geometry cases, credits\n";
}
