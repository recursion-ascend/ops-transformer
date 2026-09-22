#pragma once
#include <algorithm>
using std::min;
template <typename T> inline T Min(T a, T b) { return std::min(a, b); }
constexpr unsigned IDX_M_IDX = 0;
constexpr unsigned IDX_N_IDX = 1;
constexpr unsigned IDX_K_IDX = 2;
