/**
 * \file aclnn_mega_moe.h
 * \brief 本地手写声明：生成的算子包汇总头 (aclnn_ops_transformer_custom.h) 漏掉了
 *        aclnnMegaMoe / aclnnMegaMoeGetWorkspaceSize 的声明，符号在 libcust_opapi.so 中存在。
 *        签名取自 op_api/aclnn_mega_moe.cpp。仅供 example / msprof 采集编译使用。
 */
#ifndef ACLNN_MEGA_MOE_H_
#define ACLNN_MEGA_MOE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMegaMoeGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2, const aclTensorList *weightScales1Optional,
    const aclTensorList *weightScales2Optional, const aclTensorList *bias1Optional, const aclTensorList *bias2Optional,
    const aclTensor *xActiveMaskOptional, const aclTensorList *sharedWeight1Optional,
    const aclTensorList *sharedWeight2Optional, const aclTensorList *sharedWeightScales1Optional,
    const aclTensorList *sharedWeightScales2Optional, const aclTensorList *sharedBias1Optional,
    const aclTensorList *sharedBias2Optional, const aclTensor *maskBufferOptional, int64_t moeExpertNum,
    int64_t epWorldSize, int64_t cclBufferSize, int64_t maxRecvTokenNum, int64_t dispatchQuantMode,
    int64_t dispatchQuantOutDtype, int64_t combineQuantMode, const char *commAlg, int64_t numMaxTokensPerRank,
    const char *activation, const aclFloatArray *activationParams, int64_t topoType, int64_t rankNumPerServer,
    int64_t topkWeightsType, aclTensor *yOut, aclTensor *expertTokenNumsOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnMegaMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_MEGA_MOE_H_
