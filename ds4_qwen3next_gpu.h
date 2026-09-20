#ifndef DS4_QWEN3NEXT_GPU_H
#define DS4_QWEN3NEXT_GPU_H
#include <stdint.h>
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ds4_qwen3next_replay ds4_qwen3next_replay;
/* Session-owned, single-device segments. begin: -2 error, -1 eager,
 * 0 capturing, 1 replayed. end: -1 launch error, 0 discarded (run eager
 * if the encode succeeded), 1 launched. Always end an active capture,
 * including failed encodes. Position zero stays eager to reset state. */
int ds4_gpu_qwen3next_replay_begin(ds4_qwen3next_replay **replay, uint32_t pos);
int ds4_gpu_qwen3next_replay_end(ds4_qwen3next_replay *replay, int commit);
void ds4_gpu_qwen3next_replay_free(ds4_qwen3next_replay *replay);
/* Independent projections of the same activation, submitted together. */
typedef struct {
    ds4_gpu_tensor *out;
    uint64_t offset;
    uint32_t type, rows;
} ds4_qwen3next_projection;
int ds4_gpu_qwen3next_project(const ds4_qwen3next_projection *projections, uint32_t count,
    const ds4_gpu_tensor *x, const void *map, uint64_t size, uint32_t columns);
/* CPU-compatible greedy selection. scratch holds 256 float/ID pairs. */
int ds4_gpu_qwen3next_argmax(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
    const ds4_gpu_tensor *logits, uint32_t vocab);
int ds4_gpu_qwen3next_embed(ds4_gpu_tensor *out, const void *map, uint64_t size,
    uint64_t offset, uint32_t type, uint32_t dim, uint32_t token);
/* Fixed 2048-wide RMSNorm. Optional add updates x before normalization. */
int ds4_gpu_qwen3next_norm(ds4_gpu_tensor *out, ds4_gpu_tensor *x,
    const ds4_gpu_tensor *add, const void *map, uint64_t size, uint64_t offset, float eps);
int ds4_gpu_qwen3next_moe_residual(ds4_gpu_tensor *x, const ds4_gpu_tensor *part,
    const ds4_gpu_tensor *weights, const ds4_gpu_tensor *shared_gate);
/* Fixed 512-expert/top-10, 512-to-2048 down projection and residual. part is
 * fallback scratch; mid must be distinct from the residual x. */
int ds4_gpu_qwen3next_moe_down_residual(ds4_gpu_tensor *x, ds4_gpu_tensor *part,
    const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
    const ds4_gpu_tensor *gate, const void *map, uint64_t size, uint64_t offset, uint32_t type,
    uint64_t shared_offset, uint32_t shared_type);
int ds4_gpu_qwen3next_attention(ds4_gpu_tensor *out, ds4_gpu_tensor *q, const ds4_gpu_tensor *qg,
    const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
    ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, ds4_gpu_tensor *partial,
    const void *map, uint64_t size, uint64_t qnorm, uint64_t knorm,
    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t pos,
    uint32_t cap, uint32_t nrot, float eps, float rope_base);
/* Unpack the legacy grouped QKVZ and beta/alpha projections. With packed=0,
 * qkv/z were projected separately and only beta/alpha need unpacking. */
int ds4_gpu_qwen3next_unpack(ds4_gpu_tensor *qkv, ds4_gpu_tensor *z,
    ds4_gpu_tensor *a, ds4_gpu_tensor *b, const ds4_gpu_tensor *qkvz,
    const ds4_gpu_tensor *ba, int packed);
/* Fused unpack, four-tap SiLU convolution, Q/K normalization and decay. */
int ds4_gpu_qwen3next_prepare_recurrent(ds4_gpu_tensor *qkv, ds4_gpu_tensor *z,
    ds4_gpu_tensor *a, ds4_gpu_tensor *b, const ds4_gpu_tensor *qkvz,
    const ds4_gpu_tensor *ba, int packed, ds4_gpu_tensor *history,
    const void *map, uint64_t size, uint64_t conv, uint64_t decay, uint64_t bias);
int ds4_gpu_qwen3next_recurrent(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
    const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
    const ds4_gpu_tensor *z, const void *map, uint64_t size, uint64_t norm, float eps);
#ifdef __cplusplus
}
#endif
#endif
