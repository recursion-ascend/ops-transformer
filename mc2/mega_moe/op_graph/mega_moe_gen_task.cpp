/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_gen_task.cpp
 * \brief 注册 MegaMoe 的 KFC 通信 GenTask，使框架/单算子执行时随计算 kernel 一起下发
 *        aicpu kfc server，服务 Dispatch/Combine 的跨卡通信。缺失此注册会导致 kernel
 *        永远等待通信 server → stream sync 超时。参照 moe_distribute_dispatch 实现。
 */
#include <vector>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"

namespace ops {
ge::Status MegaMoeCalcParamFunc(gert::ExeResGenerationContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "MegaMoe do general calc param");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status MegaMoeGenTaskFunc(const gert::ExeResGenerationContext *context,
                              std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (IsTargetPlatformSocVersion(nodeName, PLATFORM_A2)) {
        OPS_LOG_D(context->GetNodeName(), "MegaMoe do A2 gen task");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    OPS_LOG_D(context->GetNodeName(), "MegaMoe do A3/A5 gen task");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

IMPL_OP(MegaMoe)
    .CalcOpParam(MegaMoeCalcParamFunc)
    .GenerateTask(MegaMoeGenTaskFunc);
} // namespace ops
