// Host-only geometry stub: does NOT emulate Ascend synchronization or memory.
#pragma once
#include <tuple>
#include <cstdint>
#define __aicore__
namespace AscendC {
namespace Std {
using std::get;
inline int64_t ceil_division(int64_t a, int64_t b) { return a / b + (a % b != 0); }
}
namespace Te {
template <typename... T> using Shape = std::tuple<T...>;
template <typename... T> using Coord = std::tuple<T...>;
}
}
