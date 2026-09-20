/* Included after ds4_qwen4_cuda.cuh: reuse its checked weight residency and
 * quantization helpers, but keep Qwen3-Next gating and recurrent head ordering separate from Qwen4. */
#include "ds4_qwen3next_gpu.h"
struct ds4_qwen3next_replay {
    cudaStream_t stream;
    cudaGraphExec_t exec;
    unsigned *position;
    unsigned window;
    int device;
    bool disabled;
};
static ds4_qwen3next_replay *qwen3next_recording;

namespace qwen3next_cuda {
__global__ void set_position(unsigned *out, unsigned pos) { *out = pos; }
}

extern "C" void ds4_gpu_qwen3next_replay_free(ds4_qwen3next_replay *r) {
    if (!r) return;
    WITH_DEVICE(r->device) {
        if (r->stream) (void)cudaStreamSynchronize(r->stream);
        if (r->exec) (void)cudaGraphExecDestroy(r->exec);
        if (r->stream) (void)cudaStreamDestroy(r->stream);
        if (r->position) (void)cudaFree(r->position);
    }
    free(r);
}

extern "C" int ds4_gpu_qwen3next_replay_begin(ds4_qwen3next_replay **slot, uint32_t pos) {
    const char *env = getenv("DS4_QWEN3NEXT_GRAPHS");
    if (!pos || (env && !strcmp(env, "0"))) return -1;
    if (!slot || qwen3next_recording || g_decode_graph_capturing) return -2;
    if (!*slot) {
        ds4_qwen3next_replay *r = (ds4_qwen3next_replay *)calloc(1, sizeof(*r));
        if (!r) return -1;
        if (!cuda_ok(cudaGetDevice(&r->device), "Next graph device")) { free(r); return -2; }
        *slot = r;
        /* A blocking stream preserves ordering with eager embedding,
         * cross-device transfers, and final readback on the legacy stream. */
        if (!cuda_ok(cudaStreamCreate(&r->stream), "Next graph stream") ||
            !cuda_ok(cudaMalloc(&r->position, sizeof(unsigned)), "Next graph position")) {
            r->disabled = true;
            (void)cudaGetLastError();
            return -1;
        }
    }
    ds4_qwen3next_replay *r = *slot;
    if (r->disabled) return -1;
    qwen3next_cuda::set_position<<<1,1,0,r->stream>>>(r->position, pos);
    if (!cuda_ok(cudaGetLastError(), "Next graph position update")) return -2;
    if (r->exec && r->window == pos / 256) {
        return cuda_ok(cudaGraphLaunch(r->exec, r->stream), "Next graph replay") ? 1 : -2;
    }
    if (r->exec) {
        (void)cudaGraphExecDestroy(r->exec);
        r->exec = NULL;
    }
    r->window = pos / 256;
    if (!cuda_ok(cudaStreamBeginCapture(r->stream, cudaStreamCaptureModeThreadLocal), "Next graph capture")) {
        r->disabled = true;
        (void)cudaGetLastError();
        return -1;
    }
    qwen3next_recording = r;
    g_qwen3next_capture_stream = r->stream;
    return 0;
}

extern "C" int ds4_gpu_qwen3next_replay_end(ds4_qwen3next_replay *r, int commit) {
    if (!r || qwen3next_recording != r) return -1;
    qwen3next_recording = NULL;
    g_qwen3next_capture_stream = NULL;
    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture(r->stream, &graph);
    bool ok = cuda_ok(err, "Next graph end") && graph && commit;
    if (ok) ok = cuda_ok(cudaGraphInstantiate(&r->exec, graph, NULL, NULL, 0), "Next graph instantiate");
    if (graph) (void)cudaGraphDestroy(graph);
    if (!ok) {
        (void)cudaGetLastError();
        r->disabled = true;
        return 0;
    }
    return cuda_ok(cudaGraphLaunch(r->exec, r->stream), "Next graph first launch") ? 1 : -1;
}

namespace qwen3next_cuda {
/* Match sample_argmax: NaNs and values below -1e30 never win, ties use the
 * lowest ID, +infinity is selectable, and an empty winner remains token 0. */
__device__ __forceinline__ unsigned argmax_key(float value) {
    unsigned bits = __float_as_uint(value);
    const unsigned magnitude = bits&0x7fffffff;
    if (magnitude>0x7f800000) return 0; /* NaN cannot win. */
    if (!magnitude) bits=0; /* Signed zeros tie. */
    return bits&0x80000000 ? ~bits : bits^0x80000000;
}
__device__ void argmax_block(unsigned &best, unsigned &id) {
    __shared__ unsigned values[8];
    __shared__ unsigned ids[8];
    const unsigned lane = threadIdx.x&31, warp = threadIdx.x/32;
    for (unsigned offset=16;offset;offset/=2) {
        const unsigned v = __shfl_xor_sync(0xffffffff,best,offset);
        const unsigned i = __shfl_xor_sync(0xffffffff,id,offset);
        if (v>best || (v==best && i<id)) { best=v; id=(unsigned)i; }
    }
    if (!lane) { values[warp]=best; ids[warp]=id; }
    __syncthreads();
    if (!warp) {
        best = lane<8 ? values[lane] : argmax_key(-1e30f);
        id = lane<8 ? ids[lane] : 0;
        for (unsigned offset=16;offset;offset/=2) {
            const unsigned v = __shfl_xor_sync(0xffffffff,best,offset);
            const unsigned i = __shfl_xor_sync(0xffffffff,id,offset);
            if (v>best || (v==best && i<id)) { best=v; id=(unsigned)i; }
        }
    }
}
__global__ void argmax_parts(unsigned *values, unsigned *ids, const float *logits, unsigned n) {
    unsigned best=argmax_key(-1e30f);
    unsigned id=0;
    for (uint64_t i=(uint64_t)blockIdx.x*256+threadIdx.x;i<n;i+=(uint64_t)gridDim.x*256) {
        const unsigned v=argmax_key(logits[i]);
        if (v>best || (v==best && i<id)) { best=v; id=(unsigned)i; }
    }
    argmax_block(best,id);
    if (!threadIdx.x) { values[blockIdx.x]=best; ids[blockIdx.x]=id; }
}
__global__ void argmax_finish(unsigned *out, const unsigned *values, const unsigned *ids, unsigned n) {
    unsigned best=threadIdx.x<n ? values[threadIdx.x] : argmax_key(-1e30f);
    unsigned id=threadIdx.x<n ? ids[threadIdx.x] : 0;
    argmax_block(best,id);
    if (!threadIdx.x) *out=id;
}
/* Adjacent columns share aligned Q5 bytes and float2 activation loads.
 * Even/odd lanes accumulate independently; their four shuffle stages precede
 * the final add, reproducing the full warp's XOR 16,8,4,2,1 tree exactly. */
__device__ __forceinline__ float q5_dot_pair(const char *row, const float *x, unsigned n) {
    const unsigned lane = threadIdx.x & 15;
    float acc[2] = {0,0};
    for (unsigned base = 0; base < n; base += 256) {
        const uint8_t *b = (const uint8_t *)row + (base/256)*176;
        const float d = __half2float(*(const __half *)b);
        const float dm = __half2float(*(const __half *)(b+2));
        const unsigned p0 = *(const unsigned *)(b+4), p1 = *(const unsigned *)(b+8), p2 = *(const unsigned *)(b+12);
        const unsigned high = *(const uint16_t *)(b+16+lane*2);
        #pragma unroll
        for (unsigned g = 0; g < 8; g++) {
            const unsigned sc = g < 4 ? (p0>>(g*8))&63 : ((p2>>((g-4)*8))&15) | (((p0>>((g-4)*8+6))&3)<<4);
            const unsigned mn = g < 4 ? (p1>>(g*8))&63 : ((p2>>((g-4)*8+4))&15) | (((p1>>((g-4)*8+6))&3)<<4);
            const unsigned low = *(const uint16_t *)(b+48+(g/2)*32+lane*2);
            const float2 xv = *(const float2 *)(x+base+g*32+lane*2);
            #pragma unroll
            for (unsigned j = 0; j < 2; j++) {
                const unsigned q = ((low>>((g&1)*4+j*8))&15) | (((high>>(g+j*8))&1)<<4);
                const float v = d*sc*q-dm*mn;
                acc[j] += v*(j ? xv.y : xv.x);
            }
        }
    }
    const unsigned mask=0xffffu << (threadIdx.x&16);
    #pragma unroll
    for (unsigned stride=8;stride;stride/=2) {
        acc[0] += __shfl_xor_sync(mask,acc[0],stride,16);
        acc[1] += __shfl_xor_sync(mask,acc[1],stride,16);
    }
    return acc[0]+acc[1];
}

/* One half-warp per expert keeps the original dot reduction. A block owns
 * all eleven contributions to one residual element, avoiding the temporary
 * projection write/read and a separate residual launch. */
__global__ void down_residual_q5(float *out, const float *mid, const int *selected,
        const float *weights, const float *gate, const char *w, const char *shared_w,
        unsigned NE, unsigned K, unsigned M, uint64_t rb) {
    const unsigned row = blockIdx.x, slot = threadIdx.x/16, lane = threadIdx.x&15;
    __shared__ float part[11];
    float v = 0;
    if (slot == 10) v = q5_dot_pair(shared_w+row*rb,mid+slot*K,K);
    else {
        const int e = selected[slot];
        if (e >= 0 && (unsigned)e < NE)
            v = q5_dot_pair(w+((uint64_t)e*M+row)*rb,mid+slot*K,K);
    }
    if (!lane) part[slot] = v;
    __syncthreads();
    if (!threadIdx.x) {
        float acc = 0;
        for (unsigned i = 0; i < 10; i++) acc += weights[i]*part[i];
        acc += qwen4_cuda::sigmoid(gate[0])*part[10];
        out[row] += acc;
    }
}

struct projections {
    float *out[3];
    const char *weight[3];
    uint64_t stride[3];
    unsigned type[3], rows[3], blocks[3];
};
template<unsigned TYPE, bool HALF = false>
__device__ void project_rows(float *out, const char *weight, uint64_t stride,
        unsigned type, unsigned rows, unsigned block, const float *x, unsigned K) {
    const unsigned row = block*(HALF ? 8 : 4) + threadIdx.x/(HALF ? 16 : 32);
    if (row >= rows) return;
    const char *w = weight + row*stride;
    float v = 0;
    if (HALF) v = q5_dot_pair(w,x,K);
    else if (TYPE != UINT_MAX) v = qwen4_cuda::dot<TYPE,false,TYPE == 13>(w,x,K);
    else switch (type) {
    case 0: v = qwen4_cuda::dot<0>(w,x,K); break;
    case 1: v = qwen4_cuda::dot<1>(w,x,K); break;
    case 13: v = qwen4_cuda::dot<13,false,true>(w,x,K); break;
    case 269: v = qwen4_cuda::dot<13>(w,x,K); break;
    case 14: v = qwen4_cuda::dot<14>(w,x,K); break;
    }
    if (!(threadIdx.x&(HALF ? 15 : 31))) out[row] = v;
}
/* Select each descriptor field at a constant parameter index, avoiding
 * dependent array indexing in every thread of the grouped projection. */
template<unsigned TYPE, bool HALF = false>
__global__ void project(projections p, const float *x, unsigned K) {
    unsigned block = blockIdx.x;
    if (block < p.blocks[0])
        project_rows<TYPE,HALF>(p.out[0],p.weight[0],p.stride[0],p.type[0],p.rows[0],block,x,K);
    else if ((block -= p.blocks[0]) < p.blocks[1])
        project_rows<TYPE,HALF>(p.out[1],p.weight[1],p.stride[1],p.type[1],p.rows[1],block,x,K);
    else
        project_rows<TYPE,HALF>(p.out[2],p.weight[2],p.stride[2],p.type[2],p.rows[2],block-p.blocks[1],x,K);
}
__global__ void moe_residual(float *x, const float *part, const float *weights, const float *gate) {
    const unsigned d = blockIdx.x*256 + threadIdx.x;
    float v = 0;
    for (unsigned s = 0; s < 10; s++) v += weights[s] * part[s*2048 + d];
    v += qwen4_cuda::sigmoid(gate[0]) * part[10*2048 + d];
    x[d] += v;
}
template<bool ADD>
__global__ void norm(float *out, float *x, const float *add, const float *weight, float eps) {
    const unsigned tid = threadIdx.x;
    float values[8], gamma[8], ss = 0;
    __shared__ float partial[256];
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) gamma[i] = weight[tid+i*256];
    /* Retain the generic 256-thread norm's per-thread sum and reduction
     * tree. Registers reuse residuals and expose the eight loads together. */
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        const unsigned d = tid + i*256;
        values[i] = x[d];
        if (ADD) { values[i] += add[d]; x[d] = values[i]; }
        ss += values[i] * values[i];
    }
    const float scale = rsqrtf(block_sum_f32_256(ss, partial) / 2048.0f + eps);
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        const unsigned d = tid + i*256;
        out[d] = values[i] * scale * gamma[i];
    }
}
__global__ void embed(float *out, const char *row, unsigned type, unsigned D) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
#define Q3_VALUE(T) case T: out[i] = qwen4_cuda::value<T>(row, i); break
    switch (type) {
        Q3_VALUE(0); Q3_VALUE(1); Q3_VALUE(2); Q3_VALUE(8); Q3_VALUE(10);
        Q3_VALUE(12); Q3_VALUE(13); Q3_VALUE(14); Q3_VALUE(16); Q3_VALUE(30); Q3_VALUE(39);
    }
#undef Q3_VALUE
}
__global__ void prepare(float *q, const float *qg, const float *k, const float *v, __half *kc, __half *vc,
    const float *gq, const float *gk, unsigned H, unsigned HK, unsigned D,
    unsigned pos, unsigned nrot, float eps, float base, const unsigned *position) {
    if (position) pos = *position;
    unsigned h = blockIdx.x, d = threadIdx.x;
    bool query = h < H;
    unsigned kh = query ? h : h - H;
    const float *src = query ? qg + (uint64_t)kh * D * 2 : k + (uint64_t)kh * D;
    __shared__ float row[256], red[32];
    float a = d < D ? src[d] : 0;
    float inv = rsqrtf(qwen4_cuda::block_sum(a * a, red) / D + eps);
    if (d < D) row[d] = a * inv * (query ? gq[d] : gk[d]);
    __syncthreads();
    /* NeoX rotates the two halves of the rotary prefix, leaving the rest unchanged. */
    if (d < nrot / 2) {
        float angle = (float)pos * powf(base, -2.0f * d / nrot);
        float c = cosf(angle), s = sinf(angle), x = row[d], y = row[d + nrot / 2];
        row[d] = x * c - y * s;
        row[d + nrot / 2] = x * s + y * c;
    }
    __syncthreads();
    if (d < D) {
        if (query) q[(uint64_t)kh * D + d] = row[d];
        else {
            uint64_t dst = ((uint64_t)pos * HK + kh) * D + d;
            kc[dst] = __float2half_rn(row[d]);
            vc[dst] = __float2half_rn(v[(uint64_t)kh * D + d]);
        }
    }
}
/* Split the causal history into 256-key partitions. Each block maintains an
 * online softmax; merge uses the partition maxima and denominators. No
 * context-sized score matrix or host round trip is needed. */
__global__ void attention_parts(float *part, const float *q, const __half *kc,
    const __half *vc, unsigned H, unsigned HK, unsigned D, unsigned pos, unsigned splits,
    const unsigned *position) {
    if (position) pos = *position;
    unsigned h = blockIdx.x, split = blockIdx.y, d = threadIdx.x;
    unsigned kh = h / (H / HK), end = min(pos + 1, (split + 1) * 256u);
    __shared__ float scores[256], rescale[256], probability[256], warp_max[8];
    /* Calculate independent key scores across eight warps, then walk the
     * cached scores without a block barrier at every key. Each warp still
     * sums the same eight 32-wide groups in the original order, and the
     * online softmax/value accumulation below keeps its original order. */
    const unsigned lane = d & 31, warp = d / 32;
    float query[8];
    #pragma unroll
    for (unsigned g = 0; g < 8; g++) query[g] = q[(uint64_t)h * D + g*32 + lane];
    for (unsigned t = split*256 + warp; t < end; t += 8) {
        float products[8];
        #pragma unroll
        for (unsigned g = 0; g < 8; g++)
            products[g] = query[g] * __half2float(kc[((uint64_t)t*HK + kh)*D + g*32 + lane]);
        /* Interleave independent shuffle chains without changing any
         * group's reduction tree or the final eight-group sum. */
        #pragma unroll
        for (unsigned stride = 16; stride; stride >>= 1) {
            #pragma unroll
            for (unsigned g = 0; g < 8; g++)
                products[g] += __shfl_xor_sync(0xffffffff, products[g], stride);
        }
        float dot = 0;
        #pragma unroll
        for (unsigned g = 0; g < 8; g++) dot += products[g];
        if (!lane) scores[t-split*256] = dot * rsqrtf((float)D);
    }
    __syncthreads();
    /* A prefix maximum gives exactly the online softmax's running maximum.
     * Compute its exponentials once per key, instead of in every value lane
     * at every iteration. Keep the ordered accumulator/denominator FMAs. */
    const unsigned count = end - split*256;
    const float score = d < count ? scores[d] : -FLT_MAX;
    float maximum = score;
    for (unsigned offset = 1; offset < 32; offset *= 2) {
        const float other = __shfl_up_sync(0xffffffff, maximum, offset);
        if (lane >= offset) maximum = fmaxf(maximum, other);
    }
    if (lane == 31) warp_max[warp] = maximum;
    __syncthreads();
    float before_warp = -FLT_MAX;
    for (unsigned w = 0; w < warp; w++) before_warp = fmaxf(before_warp, warp_max[w]);
    maximum = fmaxf(maximum, before_warp);
    float previous = __shfl_up_sync(0xffffffff, maximum, 1);
    if (!lane) previous = before_warp;
    scores[d] = maximum;
    rescale[d] = expf(previous - maximum);
    probability[d] = expf(score - maximum);
    __syncthreads();
    float acc = 0, denom = 0;
    for (unsigned t = split * 256; t < end; t++) {
        uint64_t idx = ((uint64_t)t * HK + kh) * D + d;
        float a = rescale[t-split*256], b = probability[t-split*256];
        acc = acc * a + (d < D ? __half2float(vc[idx]) * b : 0);
        denom = denom * a + b;
    }
    float *dst = part + ((uint64_t)h * splits + split) * (D + 2);
    if (d < D) dst[d] = acc;
    if (!d) { dst[D] = scores[count-1]; dst[D + 1] = denom; }
}
__global__ void attention_merge(float *out, const float *part, const float *qg, unsigned D, unsigned splits) {
    unsigned h = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    const float *p = part + (uint64_t)h * splits * (D + 2);
    float maximum = -FLT_MAX;
    for (unsigned s = 0; s < splits; s++) maximum = fmaxf(maximum, p[(uint64_t)s * (D + 2) + D]);
    float acc = 0, denom = 0;
    for (unsigned s = 0; s < splits; s++) {
        const float *row = p + (uint64_t)s * (D + 2);
        float a = expf(row[D] - maximum);
        acc += row[d] * a;
        denom += row[D + 1] * a;
    }
    out[(uint64_t)h * D + d] = acc / denom * qwen4_cuda::sigmoid(qg[(uint64_t)h * D * 2 + D + d]);
}
}
extern "C" int ds4_gpu_qwen3next_argmax(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
    const ds4_gpu_tensor *logits, uint32_t n) {
    using namespace qwen4_cuda;
    if (!n || !tensor(out,4) || !tensor(scratch,256*8) || !tensor(logits,(uint64_t)n*4)) return 0;
    const unsigned blocks = (unsigned)std::min((uint64_t)256,((uint64_t)n+255)/256);
    unsigned *values = (unsigned *)scratch->ptr;
    unsigned *ids = values+256;
    qwen3next_cuda::argmax_parts<<<blocks,256,0,cuda_decode_stream()>>>(values,ids,(const float *)logits->ptr,n);
    qwen3next_cuda::argmax_finish<<<1,256,0,cuda_decode_stream()>>>((unsigned *)out->ptr,values,ids,blocks);
    return launched();
}
extern "C" int ds4_gpu_qwen3next_project(const ds4_qwen3next_projection *items, uint32_t count,
    const ds4_gpu_tensor *x, const void *map, uint64_t size, uint32_t K) {
    using namespace qwen4_cuda;
    if (!items || !count || count > 3 || !K || !tensor(x,(uint64_t)K*4)) return 0;
    bool supported = true;
    for (unsigned i = 0; i < count; i++)
        if (items[i].type != 0 && items[i].type != 1 && items[i].type != 13 && items[i].type != 14) supported = false;
    if (!supported) {
        for (unsigned i = 0; i < count; i++)
            if (!ds4_gpu_qwen4_dense_mm_tensor(items[i].out,x,map,size,items[i].offset,items[i].type,1,K,items[i].rows)) return 0;
        return 1;
    }
    qwen3next_cuda::projections p = {};
    unsigned blocks = 0;
    for (unsigned i = 0; i < count; i++) {
        const ds4_qwen3next_projection *v = &items[i];
        p.stride[i] = row_bytes(v->type,K);
        if (!v->rows || !p.stride[i] || !tensor(v->out,(uint64_t)v->rows*4)) return 0;
        p.weight[i] = weight(map,size,v->offset,p.stride[i]*v->rows);
        if (!p.weight[i]) return 0;
        p.out[i] = (float *)v->out->ptr;
        p.type[i] = v->type == 13 && ((uintptr_t)p.weight[i]&3) ? 269 : v->type; p.rows[i] = v->rows; p.blocks[i] = ((uint64_t)v->rows+3)/4;
        blocks += p.blocks[i];
    }
    unsigned common = p.type[0];
    for (unsigned i = 1; i < count; i++) if (p.type[i] != common) common = UINT_MAX;
    if (common == 13 && K == 2048 && p.rows[0] == 12288 && !((uintptr_t)x->ptr&7)) {
        /* The wide legacy QKVZ shape benefits from two ordered accumulators
         * per lane; smaller matrices retain the full-warp implementation. */
        blocks = 0;
        for (unsigned i=0;i<count;i++) { p.blocks[i]=((uint64_t)p.rows[i]+7)/8; blocks+=p.blocks[i]; }
        qwen3next_cuda::project<13,true><<<blocks,128,0,cuda_decode_stream()>>>(p,(const float *)x->ptr,K);
    } else if (common == 13) qwen3next_cuda::project<13><<<blocks,128,0,cuda_decode_stream()>>>(p,(const float *)x->ptr,K);
    else if (common == 14) qwen3next_cuda::project<14><<<blocks,128,0,cuda_decode_stream()>>>(p,(const float *)x->ptr,K);
    else qwen3next_cuda::project<UINT_MAX><<<blocks,128,0,cuda_decode_stream()>>>(p,(const float *)x->ptr,K);
    return launched();
}
extern "C" int ds4_gpu_qwen3next_moe_residual(ds4_gpu_tensor *x, const ds4_gpu_tensor *part,
    const ds4_gpu_tensor *weights, const ds4_gpu_tensor *gate) {
    using namespace qwen4_cuda;
    if (!tensor(x, 2048*4) || !tensor(part, 11*2048*4) || !tensor(weights, 10*4) || !tensor(gate, 4)) return 0;
    qwen3next_cuda::moe_residual<<<8,256,0,cuda_decode_stream()>>>((float *)x->ptr,
        (const float *)part->ptr, (const float *)weights->ptr, (const float *)gate->ptr);
    return launched();
}
extern "C" int ds4_gpu_qwen3next_moe_down_residual(ds4_gpu_tensor *x, ds4_gpu_tensor *part,
    const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
    const ds4_gpu_tensor *gate, const void *map, uint64_t size, uint64_t off, uint32_t type,
    uint64_t shared_off, uint32_t shared_type) {
    using namespace qwen4_cuda;
    if (!tensor(x,2048*4) || !tensor(part,11*2048*4) || !tensor(mid,11*512*4) ||
        !tensor(selected,10*4) || !tensor(weights,10*4) || !tensor(gate,4)) return 0;
    if (type == 13 && shared_type == 13 && !((uintptr_t)mid->ptr&7)) {
        const char *w = weight(map,size,off,(uint64_t)512*2048*352);
        const char *sw = weight(map,size,shared_off,2048*352);
        if (!w || !sw) return 0;
        if (!(((uintptr_t)w | (uintptr_t)sw)&3)) {
            qwen3next_cuda::down_residual_q5<<<2048,176,0,cuda_decode_stream()>>>((float *)x->ptr,
                (const float *)mid->ptr,(const int *)selected->ptr,(const float *)weights->ptr,
                (const float *)gate->ptr,w,sw,512,512,2048,352);
            return launched();
        }
    }
    return ds4_gpu_qwen4_moe_down_tensor(part,mid,selected,map,size,off,type,512,1,10,512,2048,
               shared_off,shared_type) && ds4_gpu_qwen3next_moe_residual(x,part,weights,gate);
}
extern "C" int ds4_gpu_qwen3next_norm(ds4_gpu_tensor *out, ds4_gpu_tensor *x,
    const ds4_gpu_tensor *add, const void *map, uint64_t size, uint64_t off, float eps) {
    using namespace qwen4_cuda;
    if (!tensor(out, 2048*4) || !tensor(x, 2048*4) || (add && !tensor(add, 2048*4))) return 0;
    const float *w = (const float *)weight(map, size, off, 2048*4);
    if (!w) return 0;
    if (add) qwen3next_cuda::norm<true><<<1,256,0,cuda_decode_stream()>>>((float *)out->ptr, (float *)x->ptr,
        (const float *)add->ptr, w, eps);
    else qwen3next_cuda::norm<false><<<1,256,0,cuda_decode_stream()>>>((float *)out->ptr, (float *)x->ptr, NULL, w, eps);
    return launched();
}
extern "C" int ds4_gpu_qwen3next_embed(ds4_gpu_tensor *out, const void *map, uint64_t size,
    uint64_t off, uint32_t type, uint32_t D, uint32_t token) {
    using namespace qwen4_cuda;
    uint64_t rb = row_bytes(type, D);
    if (!rb || !tensor(out, (uint64_t)D * 4) || (uint64_t)token > (UINT64_MAX - off) / rb) return 0;
    const char *row = weight(map, size, off + rb * token, rb);
    if (!row) return 0;
    qwen3next_cuda::embed<<<(D+255)/256,256,0,cuda_decode_stream()>>>((float *)out->ptr, row, type, D);
    return launched();
}
extern "C" int ds4_gpu_qwen3next_attention(ds4_gpu_tensor *out, ds4_gpu_tensor *q, const ds4_gpu_tensor *qg,
    const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
    ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, ds4_gpu_tensor *partial,
    const void *map, uint64_t size, uint64_t qo, uint64_t ko,
    uint32_t H, uint32_t HK, uint32_t D, uint32_t pos, uint32_t cap, uint32_t nrot, float eps, float base) {
    using namespace qwen4_cuda;
    unsigned splits = pos / 256 + 1;
    if (!H || !HK || H % HK || D != 256 || !nrot || nrot > D || nrot % 2 || pos >= cap ||
        !tensor(qg,(uint64_t)H*D*8) || !tensor(out,(uint64_t)H*D*4) || !tensor(q,(uint64_t)H*D*4) ||
        !tensor(k,(uint64_t)HK*D*4) || !tensor(v,(uint64_t)HK*D*4) ||
        !tensor(kc,(uint64_t)cap*HK*D*2) || !tensor(vc,(uint64_t)cap*HK*D*2) ||
        !tensor(partial,(uint64_t)H*splits*(D+2)*4)) return 0;
    const float *gq = (const float *)weight(map,size,qo,D*4);
    const float *gk = (const float *)weight(map,size,ko,D*4);
    if (!gq || !gk) return 0;
    const unsigned *position = qwen3next_recording ? qwen3next_recording->position : NULL;
    qwen3next_cuda::prepare<<<H+HK,256,0,cuda_decode_stream()>>>((float *)q->ptr,
        (const float *)qg->ptr,(const float *)k->ptr,(const float *)v->ptr,(__half *)kc->ptr,(__half *)vc->ptr,
        gq,gk,H,HK,D,pos,nrot,eps,base,position);
    qwen3next_cuda::attention_parts<<<dim3(H,splits),256,0,cuda_decode_stream()>>>((float *)partial->ptr,
        (const float *)q->ptr,(const __half *)kc->ptr,(const __half *)vc->ptr,H,HK,D,pos,splits,position);
    qwen3next_cuda::attention_merge<<<H,256,0,cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)partial->ptr,(const float *)qg->ptr,D,splits);
    return launched();
}

namespace qwen3next_cuda {
/* Legacy projections are grouped by key head: [Q128,K128,V256,Z256]
 * and [beta0,beta1,alpha0,alpha1]. DeltaNet uses two consecutive value
 * heads per key head, unlike Qwen4's cyclic mapping. */
__global__ void unpack(float *qkv, float *z, float *a, float *b,
                        const float *qkvz, const float *ba, bool packed) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (packed && i < 8192) {
        unsigned h, src;
        if (i < 2048) { h = i / 128; src = h * 768 + i % 128; }
        else if (i < 4096) { h = (i - 2048) / 128; src = h * 768 + 128 + i % 128; }
        else { h = (i - 4096) / 256; src = h * 768 + 256 + (i - 4096) % 256; }
        qkv[i] = qkvz[src];
    }
    if (packed && i < 4096) z[i] = qkvz[(i / 256) * 768 + 512 + i % 256];
    if (i < 32) { b[i] = ba[(i / 2) * 4 + i % 2]; a[i] = ba[(i / 2) * 4 + 2 + i % 2]; }
}
/* A warp owns each 128-channel Q/K/V row. Four contiguous values per lane
 * preserve gdn_prep's sum/reduction order while keeping convolution results
 * in registers. History is read before shifting, even for in-place inputs. */
__global__ void prepare_recurrent(float *qkv, float *z, float *a, float *b,
        const float *qkvz, const float *ba, float *history, const float *w,
        const float *A, const float *bias, bool packed, unsigned dimension) {
    const unsigned h = blockIdx.x/4, row = blockIdx.x%4, lane = threadIdx.x;
    const unsigned first = row < 2 ? row*2048+h*128+lane*4 : 4096+h*256+(row-2)*128+lane*4;
    const unsigned src = h*768+row*128+lane*4;
    const float4 raw4 = *(const float4 *)(packed ? qkvz+src : qkv+first);
    const float4 h04 = *(const float4 *)(history+first);
    const float4 h14 = *(const float4 *)(history+8192+first);
    const float4 h24 = *(const float4 *)(history+16384+first);
    float values[4], ss = 0;
    #pragma unroll
    for (unsigned j = 0; j < 4; j++) {
        const unsigned c = first+j;
        const float raw = ((const float *)&raw4)[j];
        const float h0 = ((const float *)&h04)[j];
        const float h1 = ((const float *)&h14)[j], h2 = ((const float *)&h24)[j];
        const float4 taps = *(const float4 *)(w+c*4);
        /* The original convolution rounds its initial product before the
         * three history FMAs. Preserve that boundary when fusing unpack. */
        float v = __fmul_rn(taps.w,raw);
        v += taps.x*h0; v += taps.y*h1; v += taps.z*h2;
        values[j] = qwen4_cuda::silu(v);
        if (row < 2) ss += values[j]*values[j];
    }
    *(float4 *)(history+first) = h14;
    *(float4 *)(history+8192+first) = h24;
    *(float4 *)(history+16384+first) = raw4;
    float scale = 1;
    if (row < 2) {
        scale = rsqrtf(qwen4_cuda::sum(ss)+1e-6f);
        /* Runtime dimension retains gdn_prep's device rsqrt rounding. */
        if (!row) scale *= rsqrtf((float)dimension);
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; j++) {
        values[j] = row < 2 ? values[j]*scale : values[j];
    }
    *(float4 *)(qkv+first) = make_float4(values[0],values[1],values[2],values[3]);
    if (packed && row >= 2)
        *(float4 *)(z+h*256+(row-2)*128+lane*4) = *(const float4 *)(qkvz+h*768+512+(row-2)*128+lane*4);
    if (!row && lane < 2) {
        const unsigned v = h*2+lane;
        a[v] = expf(A[v]*qwen4_cuda::softplus(ba[h*4+2+lane]+bias[v]));
        b[v] = qwen4_cuda::sigmoid(ba[h*4+lane]);
    }
}

__global__ void recurrent(float *out, float *state, const float *qkv,
                           const float *a, const float *b) {
    const unsigned dv = blockIdx.x * 4 + threadIdx.x / 32, h = blockIdx.y;
    const unsigned lane = threadIdx.x & 31, k0 = lane * 4, kh = h / 2;
    const unsigned idx = (h * 128 + dv) * 128 + k0;
    const float *q = qkv + kh * 128 + k0, *k = q + 2048;
    float s[4], u = 0, o = 0;
    for (unsigned i = 0; i < 4; i++) { s[i] = state[idx + i] * a[h]; u += s[i] * k[i]; }
    float delta = (qkv[4096 + h * 128 + dv] - qwen4_cuda::sum(u)) * b[h];
    for (unsigned i = 0; i < 4; i++) { s[i] += k[i] * delta; state[idx + i] = s[i]; o += s[i] * q[i]; }
    o = qwen4_cuda::sum(o);
    if (!lane) out[h * 128 + dv] = o;
}
__global__ void norm_gate(float *out, const float *z, const float *w, float eps) {
    unsigned h = blockIdx.x, lane = threadIdx.x, idx = h * 128 + lane * 4;
    float ss = 0;
    for (unsigned i = 0; i < 4; i++) ss += out[idx+i] * out[idx+i];
    float inv = rsqrtf(qwen4_cuda::sum(ss) / 128 + eps);
    for (unsigned i = 0; i < 4; i++) out[idx+i] *= inv * w[lane*4+i] * qwen4_cuda::silu(z[idx+i]);
}
}
extern "C" int ds4_gpu_qwen3next_unpack(ds4_gpu_tensor *qkv, ds4_gpu_tensor *z,
    ds4_gpu_tensor *a, ds4_gpu_tensor *b, const ds4_gpu_tensor *qkvz,
    const ds4_gpu_tensor *ba, int packed) {
    using namespace qwen4_cuda;
    if (!tensor(qkv,8192*4) || !tensor(z,4096*4) || !tensor(a,32*4) ||
        !tensor(b,32*4) || !tensor(ba,64*4) || (packed && !tensor(qkvz,12288*4))) return 0;
    qwen3next_cuda::unpack<<<32,256,0,cuda_decode_stream()>>>((float *)qkv->ptr,
        (float *)z->ptr,(float *)a->ptr,(float *)b->ptr,packed ? (const float *)qkvz->ptr : NULL,
        (const float *)ba->ptr,packed != 0);
    return launched();
}
extern "C" int ds4_gpu_qwen3next_prepare_recurrent(ds4_gpu_tensor *qkv, ds4_gpu_tensor *z,
    ds4_gpu_tensor *a, ds4_gpu_tensor *b, const ds4_gpu_tensor *qkvz,
    const ds4_gpu_tensor *ba, int packed, ds4_gpu_tensor *history,
    const void *map, uint64_t size, uint64_t conv, uint64_t decay, uint64_t bias) {
    using namespace qwen4_cuda;
    if (!tensor(qkv,8192*4) || !tensor(z,4096*4) || !tensor(a,32*4) ||
        !tensor(b,32*4) || !tensor(ba,64*4) || !tensor(history,3*8192*4) ||
        (packed && !tensor(qkvz,12288*4))) return 0;
    const float *w = (const float *)weight(map,size,conv,8192*4*4);
    const float *A = (const float *)weight(map,size,decay,32*4);
    const float *dt = (const float *)weight(map,size,bias,32*4);
    if (!w || !A || !dt) return 0;
    if (((uintptr_t)qkv->ptr | (uintptr_t)z->ptr | (uintptr_t)history->ptr |
         (packed ? (uintptr_t)qkvz->ptr : 0) | (uintptr_t)w)&15) {
        return ds4_gpu_qwen3next_unpack(qkv,z,a,b,qkvz,ba,packed) &&
            ds4_gpu_qwen4_conv_stream_tensor(qkv,history,map,size,conv,1,8192,4,true) &&
            ds4_gpu_qwen4_gdn_prep_tensor(qkv,a,b,map,size,decay,bias,1,16,32,128);
    }
    qwen3next_cuda::prepare_recurrent<<<64,32,0,cuda_decode_stream()>>>((float *)qkv->ptr,
        (float *)z->ptr,(float *)a->ptr,(float *)b->ptr,packed ? (const float *)qkvz->ptr : NULL,
        (const float *)ba->ptr,(float *)history->ptr,w,A,dt,packed != 0,128);
    return launched();
}
extern "C" int ds4_gpu_qwen3next_recurrent(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
    const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
    const ds4_gpu_tensor *z, const void *map, uint64_t size, uint64_t norm, float eps) {
    using namespace qwen4_cuda;
    if (!tensor(out,4096*4) || !tensor(state,32*128*128*4) || !tensor(qkv,8192*4) ||
        !tensor(a,32*4) || !tensor(b,32*4) || !tensor(z,4096*4)) return 0;
    const float *w = (const float *)weight(map,size,norm,128*4);
    if (!w) return 0;
    qwen3next_cuda::recurrent<<<dim3(32,32),128,0,cuda_decode_stream()>>>((float *)out->ptr,
        (float *)state->ptr,(const float *)qkv->ptr,(const float *)a->ptr,(const float *)b->ptr);
    qwen3next_cuda::norm_gate<<<32,32,0,cuda_decode_stream()>>>((float *)out->ptr,(const float *)z->ptr,w,eps);
    return launched();
}
