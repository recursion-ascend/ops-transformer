# Arch35 A8W8 六层在线调度：实验实现

状态：**仅完成 CPU 策略/几何检查，尚未进行 CANN 编译、NPU 正确性或性能验证。**
默认关闭新策略。`main` 和两个原实验分支不改动。不要直接部署生产。

基线：main `a093e954a5a431b19fccf2cd92ef742c0500527c`。
整合 readiness 分支 `fca97ffa8286771a657bfee69a56e39ec4640418` 和 load-aware
分支 `237ad62353d1dc3f69c70c2ebb1b31a8ca3443bd` 的工作，修正发布协议。

## 六层与实际代码

| 层 | 实现 | 与概念方案的边界 |
|---|---|---|
| 1. Load-aware wave | `SelectRuntimeMGroupsPerWave` / `OnlinePolicy::SelectWave` | 本轮 count 表就绪后单 AIV1 发布统一正整数；只选 base 或 base/2，不扩大 host 已验证容量。skew 阈值是实验假设，不代表必然加速。 |
| 2. Group 描述符 | `WindowDesc` / `WindowCursor` | 描述专家当前 problem 内的 256-row group 连续段；offset/count 单字编码；不移动权重、Token 或专家尾块。 |
| 3. Bounded readiness | `SelectGmm1ReadyWindow` | 最多看 8 个 group，选最早 ready 连续段；frontier 未就绪时最多绕过 7 个 group，防止无界饥饿。 |
| 4. GMM1 发射与 swizzle | `Gmm1ExecGeneric` / `SelectSwizzle` | 单 AIC leader 发布调度，所有 AIC/AIV0 重放相同描述符；窗口内动态选择 1–4 的 swizzle，GMM2 保持原 swizzle。不是全局 atomic work-stealing。 |
| 5. GMM2 credit | `Gmm2AicMmadGeneric` / `CombineTokenRange` | 新增 produced/consumed 协议；12 是可调整的软件发射上限，不是原有硬件限制。不在 GMM2 内抢占切回 GMM1。 |
| 6. AIV1 仲裁 | `ProcessMoeExpertStages` | 当前 wave 已 Dispatch 后，若本 lane 的 produced-consumed >= 8，先 Combine 当前 wave，再 Dispatch 下一 wave；只在搬运排空边界切换。不是中途抢占。 |

六层仅在 A8W8 Wave、非 TopkWeightsPrefetch、非量化 Combine 的主路径联合生效。
不改变 arch22/910B，不实现 URMA/Layered 调度，不声称支持跨专家乱序或跨卡迁移。
共享专家保持原流程；量化 Combine、prefetch 路径回退原调度。

## 同步与内存协议

- adaptiveWaveControlPtr 分配一个 cache line，纳入原 flag reset 区。
  selected 值同时作为 ready 标志，不依赖“数据 store + ready store”的发布顺序。
  host tiling 和 device 使用相同 WorkspaceInfo 构造函数，必须一起重编译。
- Dispatch flag 每个 group 的 word 0 仍是 ready-row 原子计数。
  word 1 用作不可变调度描述符，按 decision ordinal 发布，不是按被选中 group 发布。
  每次至少消费一组，因此 decision 数不超过当前 problem 的 group 数；各 wave 的
  flag 基址随 expertMGroupOffset 移动，不重复使用旧 publication。
- AIC job 0 即使没有分到 tile 也必须进入描述符发布协议；因此启用时不走原 no-work 跳过路径。
- AIC/AIV0 只重排输入 group，保留原 UB ping-pong 的 wait/notify；消费者不能自行扫描 ready
  决定顺序，否则会把 tile 数据送到错误的 Activation 地址。
- 非量化 GMM2 每 AIC 的 cache line：word 0 是 produced 序号，word 1 是 consumed。
  Combine 在原 MTE3_MTE2 完成栅栏后更新 consumed；AIC 在发射前限制未消费 tile 数。
  现有 GM 输出并非 12-slot 环形缓冲，credit 仅控制生产者超前程度，可能反而降低性能。
- AIV1 每次 Dispatch 调用会排空动态搬运 ring，Combine 每 tile 排空 MTE3，再复用 UB。
  不增加全局 SyncAll，不改变跨 rank barrier。无法获得新性能数据时默认不开启。

## 启用与消融（仓库根目录运行）

```bash
python3 mc2/mega_moe/examples/configure_online_scheduler.py --preset full
bash build.sh --ophost --opapi --opkernel --ops=mega_moe --soc=ascend950 -j16
```

切换回默认：

```bash
python3 mc2/mega_moe/examples/configure_online_scheduler.py --preset baseline
# 仍须重新编译、安装并启动新进程；仅修改头文件不会改变已加载算子。
```

支持 baseline/load/ready/ready-swizzle/load-ready/credit/arbitration/full。
不要同时使用冲突的编译器 -D 覆盖配置脚本。

## 本地可复现检查

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I mc2/mega_moe/tests/standalone/stubs \
  mc2/mega_moe/tests/standalone/test_online_policy.cpp \
  -o /tmp/mega_moe_online_policy_test
/tmp/mega_moe_online_policy_test
```

测试覆盖所有 8-bit ready/done mask、空专家、256 边界、极端计数、1 万组随机
描述符重放与实际 swizzle 几何、轮转 core 分配和 credit 算术。
stubs 仅提供 tuple/shape 几何类型，不模拟设备编译、缓存一致性、event 或通信。

## 必须完成的上卡验收

1. baseline 与 full 均完成 Ascend 950 编译；分别安装到隔离环境，禁止混用 host/kernel 包。
2. 与独立 MoE reference 校验数值：均匀、单热点、多热点、空专家、1/255/256/257 行尾块；
   覆盖 interleaved/non-interleaved、ND/NZ，以及禁用策略的 quantized Combine/prefetch 回归。
3. 同一 workspace 重复至少 1000 次，改变路由分布，检查旧 flag、重复消费、漏算和死锁。
   记录 core 数，包含 leader 没分到 tile、小矩阵和零输入工作量。
4. 两卡先验证，再在可用时测试四卡及更大 EP；不能用两卡结果宣称多卡性能收益。
5. 对八个 preset 做相同输入、交替顺序的 A/B 重复测量；记录 git SHA、配置、CANN、SoC、
   dtype、shape、routing seed、各 rank latency，并以最慢 rank 的总延迟比较。
6. 使用继承的 `benchmark_readiness_aware.py --help` 查看现有 benchmark 接口；检查它的
   reference 和 dtype 是否覆盖本次测试，不能把 CPU 单测当成算子正确性验证。
7. 对 full 更慢的 case 拆分消融；未证明稳定收益的策略不要启用。没有自动在线实测回滚。

主要性能风险：AIC leader 的控制开销、权重局部性下降、过小 wave、credit 阻塞、
AIV1 延迟下一波 Dispatch。不存在对任意 Token 分布都更快的保证。
