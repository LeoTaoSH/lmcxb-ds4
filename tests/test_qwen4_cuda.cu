#include "ds4_gpu.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static void check_hc_combine() {
    constexpr uint32_t tokens = 2;
    constexpr uint32_t embd = 257;
    constexpr uint32_t hc = 4;
    constexpr uint32_t chunks = DS4_QWEN4_HC_CHUNKS;

    std::vector<float> residual((size_t)tokens * hc * embd);
    std::vector<float> block((size_t)tokens * embd);
    std::vector<float> inject((size_t)tokens * hc * chunks * hc);
    for (size_t i = 0; i < residual.size(); i++) residual[i] = ((int)(i % 29) - 14) * 0.03125f;
    for (size_t i = 0; i < block.size(); i++) block[i] = ((int)(i % 17) - 8) * 0.0625f;
    for (size_t i = 0; i < inject.size(); i++) inject[i] = ((int)(i % 13) - 6) * 0.015625f;

    std::vector<float> expected = residual;
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t s = 0; s < hc; s++) {
            float sum = 0.0f;
            const size_t base = (size_t)t * hc * chunks * hc;
            for (uint32_t src = 0; src < hc * chunks; src++) sum += inject[base + (size_t)src * hc + s];
            const float weight = 2.0f * sigmoid(sum / (float)hc);
            for (uint32_t d = 0; d < embd; d++)
                expected[((size_t)t * hc + s) * embd + d] += weight * block[(size_t)t * embd + d];
        }
    }

    ds4_gpu_tensor *d_residual = ds4_gpu_tensor_alloc(residual.size() * sizeof(float));
    ds4_gpu_tensor *d_block = ds4_gpu_tensor_alloc(block.size() * sizeof(float));
    ds4_gpu_tensor *d_inject = ds4_gpu_tensor_alloc(inject.size() * sizeof(float));
    CHECK(d_residual && d_block && d_inject);
    CHECK(ds4_gpu_tensor_write(d_residual, 0, residual.data(), residual.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(d_block, 0, block.data(), block.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(d_inject, 0, inject.data(), inject.size() * sizeof(float)));
    CHECK(ds4_gpu_qwen4_hc_combine_tensor(d_residual, d_block, d_inject, tokens, embd, hc));
    CHECK(!ds4_gpu_qwen4_hc_combine_tensor(d_residual, d_block, d_inject, tokens, embd, 8));
    CHECK(ds4_gpu_synchronize());

    std::vector<float> actual(residual.size());
    CHECK(ds4_gpu_tensor_read(d_residual, 0, actual.data(), actual.size() * sizeof(float)));
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); i++) worst = std::fmax(worst, std::fabs(actual[i] - expected[i]));
    CHECK(worst < 2e-6f);

    ds4_gpu_tensor_free(d_inject);
    ds4_gpu_tensor_free(d_block);
    ds4_gpu_tensor_free(d_residual);
    printf("Qwen3.8 CUDA hyper-connection combine PASS (worst %.9g)\n", worst);
}

static void check_conv_stream(bool apply_silu) {
    constexpr uint32_t tokens = 5;
    constexpr uint32_t channels = 7;
    constexpr uint32_t kernel = 4;
    std::vector<float> weight((size_t)channels * kernel);
    std::vector<float> input((size_t)tokens * channels);
    std::vector<float> state((size_t)(kernel - 1u) * channels);
    for (size_t i = 0; i < weight.size(); i++) weight[i] = ((int)(i % 11) - 5) * 0.046875f;
    for (size_t i = 0; i < input.size(); i++) input[i] = ((int)(i % 19) - 9) * 0.078125f;
    for (size_t i = 0; i < state.size(); i++) state[i] = ((int)(i % 7) - 3) * 0.09375f;

    std::vector<float> expected = input;
    std::vector<float> expected_state = state;
    for (uint32_t c = 0; c < channels; c++) {
        float window[3];
        for (uint32_t i = 0; i + 1u < kernel; i++) window[i] = state[(size_t)i * channels + c];
        for (uint32_t t = 0; t < tokens; t++) {
            const float raw = input[(size_t)t * channels + c];
            float sum = weight[(size_t)c * kernel + kernel - 1u] * raw;
            for (uint32_t i = 0; i + 1u < kernel; i++) sum += weight[(size_t)c * kernel + i] * window[i];
            for (uint32_t i = 0; i + 2u < kernel; i++) window[i] = window[i + 1u];
            window[kernel - 2u] = raw;
            expected[(size_t)t * channels + c] = apply_silu ? sum * sigmoid(sum) : sum;
        }
        for (uint32_t i = 0; i + 1u < kernel; i++) expected_state[(size_t)i * channels + c] = window[i];
    }

    CHECK(ds4_gpu_set_model_map(weight.data(), weight.size() * sizeof(float)));
    ds4_gpu_tensor *d_input = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    ds4_gpu_tensor *d_state = ds4_gpu_tensor_alloc(state.size() * sizeof(float));
    CHECK(d_input && d_state);
    CHECK(ds4_gpu_tensor_write(d_input, 0, input.data(), input.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(d_state, 0, state.data(), state.size() * sizeof(float)));
    CHECK(ds4_gpu_qwen4_conv_stream_tensor(d_input, d_state, weight.data(), weight.size() * sizeof(float),
                                            0, tokens, channels, kernel, apply_silu));
    CHECK(!ds4_gpu_qwen4_conv_stream_tensor(d_input, d_state, weight.data(), weight.size() * sizeof(float),
                                             0, tokens, channels, 1, apply_silu));
    CHECK(ds4_gpu_synchronize());

    std::vector<float> actual(input.size()), actual_state(state.size());
    CHECK(ds4_gpu_tensor_read(d_input, 0, actual.data(), actual.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_state, 0, actual_state.data(), actual_state.size() * sizeof(float)));
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); i++) worst = std::fmax(worst, std::fabs(actual[i] - expected[i]));
    for (size_t i = 0; i < actual_state.size(); i++)
        worst = std::fmax(worst, std::fabs(actual_state[i] - expected_state[i]));
    CHECK(worst < 2e-6f);
    ds4_gpu_tensor_free(d_state);
    ds4_gpu_tensor_free(d_input);
    printf("Qwen3.8 CUDA causal conv%s PASS (worst %.9g)\n", apply_silu ? "+SiLU" : "", worst);
}

static float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

static void check_gdn_prep() {
    constexpr uint32_t tokens = 2;
    constexpr uint32_t kh = 2;
    constexpr uint32_t vh = 3;
    constexpr uint32_t dim = 64;
    constexpr uint32_t conv_dim = (2 * kh + vh) * dim;
    std::vector<float> model(2u * vh);
    for (uint32_t i = 0; i < vh; i++) {
        model[i] = -0.25f - 0.125f * i;
        model[vh + i] = -0.2f + 0.15f * i;
    }
    std::vector<float> qkv((size_t)tokens * conv_dim);
    std::vector<float> alpha((size_t)tokens * vh), beta((size_t)tokens * vh);
    for (size_t i = 0; i < qkv.size(); i++) qkv[i] = ((int)(i % 31) - 15) * 0.03125f;
    for (size_t i = 0; i < alpha.size(); i++) {
        alpha[i] = ((int)i - 2) * 0.2f;
        beta[i] = ((int)i - 3) * 0.3f;
    }
    std::vector<float> expected_qkv = qkv, expected_alpha = alpha, expected_beta = beta;
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < kh; h++) {
            float *q = expected_qkv.data() + (size_t)t * conv_dim + (size_t)h * dim;
            float *k = q + (size_t)kh * dim;
            float qsum = 0.0f, ksum = 0.0f;
            for (uint32_t d = 0; d < dim; d++) { qsum += q[d] * q[d]; ksum += k[d] * k[d]; }
            const float qs = 1.0f / std::sqrt(qsum + 1e-6f) / std::sqrt((float)dim);
            const float ks = 1.0f / std::sqrt(ksum + 1e-6f);
            for (uint32_t d = 0; d < dim; d++) { q[d] *= qs; k[d] *= ks; }
        }
        for (uint32_t h = 0; h < vh; h++) {
            const size_t i = (size_t)t * vh + h;
            expected_alpha[i] = std::exp(model[h] * softplus(alpha[i] + model[vh + h]));
            expected_beta[i] = sigmoid(beta[i]);
        }
    }

    CHECK(ds4_gpu_set_model_map(model.data(), model.size() * sizeof(float)));
    ds4_gpu_tensor *d_qkv = ds4_gpu_tensor_alloc(qkv.size() * sizeof(float));
    ds4_gpu_tensor *d_alpha = ds4_gpu_tensor_alloc(alpha.size() * sizeof(float));
    ds4_gpu_tensor *d_beta = ds4_gpu_tensor_alloc(beta.size() * sizeof(float));
    CHECK(d_qkv && d_alpha && d_beta);
    CHECK(ds4_gpu_tensor_write(d_qkv, 0, qkv.data(), qkv.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(d_alpha, 0, alpha.data(), alpha.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(d_beta, 0, beta.data(), beta.size() * sizeof(float)));
    CHECK(ds4_gpu_qwen4_gdn_prep_tensor(d_qkv, d_alpha, d_beta, model.data(), model.size() * sizeof(float),
                                         0, vh * sizeof(float), tokens, kh, vh, dim));
    CHECK(!ds4_gpu_qwen4_gdn_prep_tensor(d_qkv, d_alpha, d_beta, model.data(), model.size() * sizeof(float),
                                          0, vh * sizeof(float), tokens, kh, vh, 48));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual_qkv(qkv.size()), actual_alpha(alpha.size()), actual_beta(beta.size());
    CHECK(ds4_gpu_tensor_read(d_qkv, 0, actual_qkv.data(), actual_qkv.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_alpha, 0, actual_alpha.data(), actual_alpha.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_beta, 0, actual_beta.data(), actual_beta.size() * sizeof(float)));
    float worst_qkv = 0.0f, worst_control = 0.0f;
    size_t worst_qkv_i = 0, worst_control_i = 0;
    for (size_t i = 0; i < actual_qkv.size(); i++) {
        const float diff = std::fabs(actual_qkv[i] - expected_qkv[i]);
        if (diff > worst_qkv) { worst_qkv = diff; worst_qkv_i = i; }
    }
    for (size_t i = 0; i < actual_alpha.size(); i++) {
        const float adiff = std::fabs(actual_alpha[i] - expected_alpha[i]);
        const float bdiff = std::fabs(actual_beta[i] - expected_beta[i]);
        if (adiff > worst_control) { worst_control = adiff; worst_control_i = 2u * i; }
        if (bdiff > worst_control) { worst_control = bdiff; worst_control_i = 2u * i + 1u; }
    }
    const float worst = std::fmax(worst_qkv, worst_control);
    if (worst >= 3e-6f)
        fprintf(stderr, "GDN prep drift: qkv=%g at %zu (got %g expected %g), control=%g at %zu\n",
                worst_qkv, worst_qkv_i, actual_qkv[worst_qkv_i], expected_qkv[worst_qkv_i],
                worst_control, worst_control_i);
    CHECK(worst < 3e-6f);
    ds4_gpu_tensor_free(d_beta);
    ds4_gpu_tensor_free(d_alpha);
    ds4_gpu_tensor_free(d_qkv);
    printf("Qwen3.8 CUDA GDN prep PASS (worst %.9g)\n", worst);
}

static void check_gdn_scan() {
    constexpr uint32_t tokens = 3;
    constexpr uint32_t kh = 2;
    constexpr uint32_t vh = 3;
    constexpr uint32_t dim = 64;
    constexpr uint32_t conv_dim = (2 * kh + vh) * dim;
    const size_t state_count = (size_t)vh * dim * dim;
    std::vector<float> qkv((size_t)tokens * conv_dim);
    std::vector<float> decay((size_t)tokens * vh), beta((size_t)tokens * vh);
    std::vector<float> initial(state_count);
    for (size_t i = 0; i < qkv.size(); i++) qkv[i] = ((int)(i % 23) - 11) * 0.0078125f;
    for (size_t i = 0; i < decay.size(); i++) {
        decay[i] = 0.82f + 0.01f * (float)(i % vh);
        beta[i] = 0.31f + 0.02f * (float)(i % 4);
    }
    for (size_t i = 0; i < initial.size(); i++) initial[i] = ((int)(i % 17) - 8) * 0.001953125f;

    std::vector<float> expected((size_t)tokens * vh * dim);
    std::vector<float> expected_state = initial, expected_snap(state_count), expected_snap2(state_count);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < vh; h++) {
            const uint32_t khead = h % kh;
            const float *q = qkv.data() + (size_t)t * conv_dim + (size_t)khead * dim;
            const float *k = q + (size_t)kh * dim;
            const float *v = qkv.data() + (size_t)t * conv_dim + (size_t)2u * kh * dim + (size_t)h * dim;
            for (uint32_t dv = 0; dv < dim; dv++) {
                float *row = expected_state.data() + ((size_t)h * dim + dv) * dim;
                float prediction = 0.0f;
                for (uint32_t dk = 0; dk < dim; dk++) {
                    row[dk] *= decay[(size_t)t * vh + h];
                    prediction += row[dk] * k[dk];
                }
                const float delta = (v[dv] - prediction) * beta[(size_t)t * vh + h];
                float value = 0.0f;
                for (uint32_t dk = 0; dk < dim; dk++) {
                    row[dk] += k[dk] * delta;
                    value += row[dk] * q[dk];
                }
                expected[((size_t)t * vh + h) * dim + dv] = value;
            }
        }
        if (t == 0) expected_snap = expected_state;
        if (t == 1) expected_snap2 = expected_state;
    }

    auto alloc_write = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    ds4_gpu_tensor *d_qkv = alloc_write(qkv);
    ds4_gpu_tensor *d_decay = alloc_write(decay);
    ds4_gpu_tensor *d_beta = alloc_write(beta);
    ds4_gpu_tensor *d_state = alloc_write(initial);
    ds4_gpu_tensor *d_serial_state = alloc_write(initial);
    ds4_gpu_tensor *d_out = ds4_gpu_tensor_alloc(expected.size() * sizeof(float));
    ds4_gpu_tensor *d_serial_out = ds4_gpu_tensor_alloc(expected.size() * sizeof(float));
    ds4_gpu_tensor *d_snap = ds4_gpu_tensor_alloc(state_count * sizeof(float));
    ds4_gpu_tensor *d_snap2 = ds4_gpu_tensor_alloc(state_count * sizeof(float));
    CHECK(d_out && d_serial_out && d_snap && d_snap2);
    CHECK(ds4_gpu_qwen4_gdn_scan_tensor(d_out, d_state, d_qkv, d_decay, d_beta,
                                         tokens, kh, vh, dim, d_snap, 0, d_snap2, 1));

    const uint64_t qkv_row_bytes = (uint64_t)conv_dim * sizeof(float);
    const uint64_t ctl_row_bytes = (uint64_t)vh * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)vh * dim * sizeof(float);
    for (uint32_t t = 0; t < tokens; t++) {
        ds4_gpu_tensor *qrow = ds4_gpu_tensor_view(d_qkv, (uint64_t)t * qkv_row_bytes, qkv_row_bytes);
        ds4_gpu_tensor *arow = ds4_gpu_tensor_view(d_decay, (uint64_t)t * ctl_row_bytes, ctl_row_bytes);
        ds4_gpu_tensor *brow = ds4_gpu_tensor_view(d_beta, (uint64_t)t * ctl_row_bytes, ctl_row_bytes);
        ds4_gpu_tensor *orow = ds4_gpu_tensor_view(d_serial_out, (uint64_t)t * out_row_bytes, out_row_bytes);
        CHECK(qrow && arow && brow && orow);
        CHECK(ds4_gpu_qwen4_gdn_scan_tensor(orow, d_serial_state, qrow, arow, brow,
                                             1, kh, vh, dim, NULL, 0, NULL, 0));
        ds4_gpu_tensor_free(orow); ds4_gpu_tensor_free(brow); ds4_gpu_tensor_free(arow); ds4_gpu_tensor_free(qrow);
    }
    CHECK(ds4_gpu_synchronize());

    std::vector<float> actual(expected.size()), serial(expected.size());
    std::vector<float> actual_state(state_count), serial_state(state_count), actual_snap(state_count), actual_snap2(state_count);
    CHECK(ds4_gpu_tensor_read(d_out, 0, actual.data(), actual.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_serial_out, 0, serial.data(), serial.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_state, 0, actual_state.data(), actual_state.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_serial_state, 0, serial_state.data(), serial_state.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_snap, 0, actual_snap.data(), actual_snap.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_snap2, 0, actual_snap2.data(), actual_snap2.size() * sizeof(float)));
    float worst = 0.0f;
    auto compare = [&worst](const std::vector<float> &got, const std::vector<float> &want) {
        CHECK(got.size() == want.size());
        for (size_t i = 0; i < got.size(); i++) worst = std::fmax(worst, std::fabs(got[i] - want[i]));
    };
    compare(actual, expected); compare(actual_state, expected_state);
    compare(actual_snap, expected_snap); compare(actual_snap2, expected_snap2);
    CHECK(worst < 2e-5f);
    CHECK(!memcmp(actual.data(), serial.data(), actual.size() * sizeof(float)));
    CHECK(!memcmp(actual_state.data(), serial_state.data(), actual_state.size() * sizeof(float)));

    ds4_gpu_tensor_free(d_snap2); ds4_gpu_tensor_free(d_snap);
    ds4_gpu_tensor_free(d_serial_out); ds4_gpu_tensor_free(d_out);
    ds4_gpu_tensor_free(d_serial_state); ds4_gpu_tensor_free(d_state);
    ds4_gpu_tensor_free(d_beta); ds4_gpu_tensor_free(d_decay); ds4_gpu_tensor_free(d_qkv);
    printf("Qwen3.8 CUDA GDN scan/snapshots PASS (worst %.9g)\n", worst);
}

static void check_gdn_out() {
    constexpr uint32_t tokens = 2, heads = 3, dim = 64;
    std::vector<float> weight(dim), output((size_t)tokens * heads * dim), gate(output.size());
    for (size_t i = 0; i < weight.size(); i++) weight[i] = 0.75f + (float)(i % 9) * 0.03125f;
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = ((int)(i % 27) - 13) * 0.0234375f;
        gate[i] = ((int)(i % 15) - 7) * 0.125f;
    }
    std::vector<float> expected = output;
    constexpr float eps = 1e-6f;
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t h = 0; h < heads; h++) {
        const size_t base = ((size_t)t * heads + h) * dim;
        float sum = 0.0f;
        for (uint32_t d = 0; d < dim; d++) sum += output[base + d] * output[base + d];
        const float scale = 1.0f / std::sqrt(sum / (float)dim + eps);
        for (uint32_t d = 0; d < dim; d++)
            expected[base + d] = output[base + d] * scale * weight[d] * sigmoid(gate[base + d]);
    }
    CHECK(ds4_gpu_set_model_map(weight.data(), weight.size() * sizeof(float)));
    ds4_gpu_tensor *d_output = ds4_gpu_tensor_alloc(output.size() * sizeof(float));
    ds4_gpu_tensor *d_gate = ds4_gpu_tensor_alloc(gate.size() * sizeof(float));
    CHECK(d_output && d_gate);
    CHECK(ds4_gpu_tensor_write(d_output, 0, output.data(), output.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(d_gate, 0, gate.data(), gate.size() * sizeof(float)));
    CHECK(ds4_gpu_qwen4_gdn_out_tensor(d_output, d_gate, weight.data(), weight.size() * sizeof(float),
                                        0, tokens, heads, dim, eps));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual(output.size());
    CHECK(ds4_gpu_tensor_read(d_output, 0, actual.data(), actual.size() * sizeof(float)));
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); i++) worst = std::fmax(worst, std::fabs(actual[i] - expected[i]));
    CHECK(worst < 3e-6f);
    ds4_gpu_tensor_free(d_gate); ds4_gpu_tensor_free(d_output);
    printf("Qwen3.8 CUDA GDN output PASS (worst %.9g)\n", worst);
}

static void check_ple_gate() {
    constexpr uint32_t tokens = 2, embd = 259, hc = 4;
    const size_t rows = (size_t)tokens * hc * embd;
    const size_t weights = (size_t)hc * embd;
    std::vector<float> residual(rows), key(rows), value((size_t)tokens * embd);
    std::vector<float> model(3u * weights);
    for (size_t i = 0; i < rows; i++) {
        residual[i] = ((int)(i % 31) - 15) * 0.01953125f;
        key[i] = ((int)(i % 23) - 11) * 0.02734375f;
    }
    for (size_t i = 0; i < value.size(); i++) value[i] = ((int)(i % 17) - 8) * 0.046875f;
    for (size_t i = 0; i < weights; i++) {
        model[i] = 0.8f + (float)(i % 7) * 0.03125f;
        model[weights + i] = 0.7f + (float)(i % 11) * 0.0234375f;
        model[2u * weights + i] = 0.9f + (float)(i % 5) * 0.0390625f;
    }
    std::vector<float> expected_gated(rows), expected_normed(rows);
    constexpr float eps = 1e-6f;
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t s = 0; s < hc; s++) {
        const size_t base = ((size_t)t * hc + s) * embd;
        const size_t wb = (size_t)s * embd;
        float sk = 0.0f, sr = 0.0f;
        for (uint32_t d = 0; d < embd; d++) {
            sk += key[base + d] * key[base + d];
            sr += residual[base + d] * residual[base + d];
        }
        const float ik = 1.0f / std::sqrt(sk / embd + eps);
        const float ir = 1.0f / std::sqrt(sr / embd + eps);
        float dot = 0.0f;
        for (uint32_t d = 0; d < embd; d++)
            dot += key[base + d] * ik * model[wb + d] *
                   residual[base + d] * ir * model[weights + wb + d];
        dot /= std::sqrt((float)embd);
        const float root = std::sqrt(std::fmax(std::fabs(dot), 1e-6f));
        const float scale = sigmoid(dot > 0.0f ? root : (dot < 0.0f ? -root : 0.0f));
        float sg = 0.0f;
        for (uint32_t d = 0; d < embd; d++) {
            const float v = scale * value[(size_t)t * embd + d];
            expected_gated[base + d] = v;
            sg += v * v;
        }
        const float ig = 1.0f / std::sqrt(sg / embd + eps);
        for (uint32_t d = 0; d < embd; d++)
            expected_normed[base + d] = expected_gated[base + d] * ig * model[2u * weights + wb + d];
    }

    auto alloc_write = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    CHECK(ds4_gpu_set_model_map(model.data(), model.size() * sizeof(float)));
    ds4_gpu_tensor *d_residual = alloc_write(residual);
    ds4_gpu_tensor *d_key = alloc_write(key);
    ds4_gpu_tensor *d_value = alloc_write(value);
    ds4_gpu_tensor *d_gated = ds4_gpu_tensor_alloc(rows * sizeof(float));
    ds4_gpu_tensor *d_normed = ds4_gpu_tensor_alloc(rows * sizeof(float));
    CHECK(d_gated && d_normed);
    CHECK(ds4_gpu_qwen4_ple_gate_tensor(d_gated, d_normed, d_residual, d_key, d_value,
                                         model.data(), model.size() * sizeof(float),
                                         0, weights * sizeof(float), 2u * weights * sizeof(float),
                                         tokens, embd, hc, eps));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual_gated(rows), actual_normed(rows);
    CHECK(ds4_gpu_tensor_read(d_gated, 0, actual_gated.data(), rows * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_normed, 0, actual_normed.data(), rows * sizeof(float)));
    float worst = 0.0f;
    for (size_t i = 0; i < rows; i++) {
        worst = std::fmax(worst, std::fabs(actual_gated[i] - expected_gated[i]));
        worst = std::fmax(worst, std::fabs(actual_normed[i] - expected_normed[i]));
    }
    CHECK(worst < 8e-6f);
    ds4_gpu_tensor_free(d_normed); ds4_gpu_tensor_free(d_gated);
    ds4_gpu_tensor_free(d_value); ds4_gpu_tensor_free(d_key); ds4_gpu_tensor_free(d_residual);
    printf("Qwen3.8 CUDA PLE gate/norm PASS (worst %.9g)\n", worst);
}

static void check_hc_norm_gate_mix() {
    constexpr uint32_t tokens = 2, embd = 64, hc = 4, injects = 4, rank = 32;
    constexpr size_t dim = (size_t)embd * hc;
    constexpr size_t gamma_off = 0;
    constexpr size_t inject_off = dim;
    constexpr size_t up_off = inject_off + injects * dim;
    std::vector<float> model(up_off + dim * rank);
    std::vector<float> residual((size_t)tokens * dim), low((size_t)tokens * rank);
    for (size_t i = 0; i < dim; i++) model[gamma_off + i] = 0.75f + (float)(i % 9) * 0.03125f;
    for (size_t i = 0; i < injects * dim; i++) model[inject_off + i] = ((int)(i % 17) - 8) * 0.015625f;
    for (size_t i = 0; i < dim * rank; i++) model[up_off + i] = ((int)(i % 23) - 11) * 0.0078125f;
    for (size_t i = 0; i < residual.size(); i++) residual[i] = ((int)(i % 29) - 14) * 0.02734375f;
    for (size_t i = 0; i < low.size(); i++) low[i] = ((int)(i % 19) - 9) * 0.0625f;

    std::vector<float> expected_xn(residual.size());
    std::vector<float> expected_part((size_t)tokens * hc * DS4_QWEN4_HC_CHUNKS * injects);
    constexpr float eps = 1e-6f;
    const uint32_t per_chunk = (embd + DS4_QWEN4_HC_CHUNKS - 1u) / DS4_QWEN4_HC_CHUNKS;
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t s = 0; s < hc; s++) {
        const size_t base = ((size_t)t * hc + s) * embd;
        float sum = 0.0f;
        for (uint32_t d = 0; d < embd; d++) sum += residual[base + d] * residual[base + d];
        const float scale = 1.0f / std::sqrt(sum / embd + eps);
        for (uint32_t d = 0; d < embd; d++)
            expected_xn[base + d] = residual[base + d] * scale * model[(size_t)s * embd + d];
        for (uint32_t c = 0; c < DS4_QWEN4_HC_CHUNKS; c++) for (uint32_t j = 0; j < injects; j++) {
            float dot = 0.0f;
            const uint32_t end = std::min(embd, (c + 1u) * per_chunk);
            for (uint32_t d = c * per_chunk; d < end; d++)
                dot += model[inject_off + (size_t)j * dim + (size_t)s * embd + d] * expected_xn[base + d];
            expected_part[(((size_t)t * hc + s) * DS4_QWEN4_HC_CHUNKS + c) * injects + j] = dot;
        }
    }
    std::vector<float> expected_mixed((size_t)tokens * embd);
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t d = 0; d < embd; d++) {
        float mixed = 0.0f;
        for (uint32_t s = 0; s < hc; s++) {
            float gate = 0.0f;
            const size_t row = ((size_t)s * embd + d) * rank;
            for (uint32_t r = 0; r < rank; r++) {
                const float x = low[(size_t)t * rank + r] / hc;
                gate += model[up_off + row + r] * x * sigmoid(x);
            }
            mixed += sigmoid(gate) * expected_xn[((size_t)t * hc + s) * embd + d];
        }
        expected_mixed[(size_t)t * embd + d] = mixed / hc;
    }

    auto alloc_write = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    CHECK(ds4_gpu_set_model_map(model.data(), model.size() * sizeof(float)));
    ds4_gpu_tensor *d_residual = alloc_write(residual), *d_low = alloc_write(low);
    ds4_gpu_tensor *d_xn = ds4_gpu_tensor_alloc(expected_xn.size() * sizeof(float));
    ds4_gpu_tensor *d_part = ds4_gpu_tensor_alloc(expected_part.size() * sizeof(float));
    ds4_gpu_tensor *d_mixed = ds4_gpu_tensor_alloc(expected_mixed.size() * sizeof(float));
    CHECK(d_xn && d_part && d_mixed);
    CHECK(ds4_gpu_qwen4_hc_norm_tensor(d_xn, d_part, d_residual, model.data(), model.size() * sizeof(float),
                                        gamma_off * sizeof(float), inject_off * sizeof(float), 0,
                                        tokens, embd, hc, injects, eps));
    CHECK(ds4_gpu_qwen4_hc_gate_mix_tensor(d_mixed, d_xn, d_low, model.data(), model.size() * sizeof(float),
                                            up_off * sizeof(float), 0, tokens, embd, hc, rank));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual_xn(expected_xn.size()), actual_part(expected_part.size()), actual_mixed(expected_mixed.size());
    CHECK(ds4_gpu_tensor_read(d_xn, 0, actual_xn.data(), actual_xn.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_part, 0, actual_part.data(), actual_part.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_mixed, 0, actual_mixed.data(), actual_mixed.size() * sizeof(float)));
    float worst = 0.0f;
    auto compare = [&worst](const std::vector<float> &got, const std::vector<float> &want) {
        for (size_t i = 0; i < got.size(); i++) worst = std::fmax(worst, std::fabs(got[i] - want[i]));
    };
    compare(actual_xn, expected_xn); compare(actual_part, expected_part); compare(actual_mixed, expected_mixed);
    CHECK(worst < 5e-6f);
    ds4_gpu_tensor_free(d_mixed); ds4_gpu_tensor_free(d_part); ds4_gpu_tensor_free(d_xn);
    ds4_gpu_tensor_free(d_low); ds4_gpu_tensor_free(d_residual);
    printf("Qwen3.8 CUDA hyper-connection norm/gate mix PASS (worst %.9g)\n", worst);
}

static void check_ple_conv(bool use_f16) {
    constexpr uint32_t tokens = 3, channels = 13, kernel = 4, dilation = 3;
    constexpr uint32_t hist_len = (kernel - 1u) * dilation;
    std::vector<float> residual((size_t)tokens * channels), gated(residual.size()), normed(residual.size());
    std::vector<float> history((size_t)hist_len * channels), weight((size_t)channels * kernel);
    for (size_t i = 0; i < residual.size(); i++) {
        residual[i] = ((int)(i % 19) - 9) * 0.03125f;
        gated[i] = ((int)(i % 13) - 6) * 0.015625f;
        normed[i] = ((int)(i % 29) - 14) * 0.0234375f;
    }
    for (size_t i = 0; i < history.size(); i++) history[i] = ((int)(i % 11) - 5) * 0.0390625f;
    for (size_t i = 0; i < weight.size(); i++) weight[i] = ((int)(i % 9) - 4) * 0.0546875f;
    std::vector<__half> half_weight;
    if (use_f16) {
        half_weight.resize(weight.size());
        for (size_t i = 0; i < weight.size(); i++) {
            half_weight[i] = __float2half(weight[i]);
            weight[i] = __half2float(half_weight[i]);
        }
    }
    std::vector<float> expected = residual, expected_history = history;
    std::vector<float> expected_snap(history.size()), expected_snap2(history.size());
    for (uint32_t c = 0; c < channels; c++) {
        float hist[hist_len];
        for (uint32_t i = 0; i < hist_len; i++) hist[i] = history[(size_t)i * channels + c];
        for (uint32_t t = 0; t < tokens; t++) {
            const size_t off = (size_t)t * channels + c;
            float sum = weight[(size_t)c * kernel + kernel - 1u] * normed[off];
            for (uint32_t k = 0; k + 1u < kernel; k++)
                sum += weight[(size_t)c * kernel + k] * hist[hist_len - (kernel - 1u - k) * dilation];
            for (uint32_t i = 0; i + 1u < hist_len; i++) hist[i] = hist[i + 1u];
            hist[hist_len - 1u] = normed[off];
            expected[off] += gated[off] + sum * sigmoid(sum);
            if (t == 0) for (uint32_t i = 0; i < hist_len; i++) expected_snap[(size_t)i * channels + c] = hist[i];
            if (t == 1) for (uint32_t i = 0; i < hist_len; i++) expected_snap2[(size_t)i * channels + c] = hist[i];
        }
        for (uint32_t i = 0; i < hist_len; i++) expected_history[(size_t)i * channels + c] = hist[i];
    }
    const void *model = use_f16 ? (const void *)half_weight.data() : (const void *)weight.data();
    const uint64_t model_size = weight.size() * (use_f16 ? sizeof(__half) : sizeof(float));
    CHECK(ds4_gpu_set_model_map(model, model_size));
    auto alloc_write = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    ds4_gpu_tensor *d_residual = alloc_write(residual), *d_gated = alloc_write(gated);
    ds4_gpu_tensor *d_normed = alloc_write(normed), *d_history = alloc_write(history);
    ds4_gpu_tensor *d_snap = ds4_gpu_tensor_alloc(history.size() * sizeof(float));
    ds4_gpu_tensor *d_snap2 = ds4_gpu_tensor_alloc(history.size() * sizeof(float));
    CHECK(d_snap && d_snap2);
    CHECK(ds4_gpu_qwen4_ple_conv_tensor(d_residual, d_gated, d_normed, d_history,
                                         model, model_size, 0, use_f16 ? 1u : 0u,
                                         tokens, channels, kernel, dilation, d_snap, 0, d_snap2, 1));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual(residual.size()), actual_history(history.size()), snap(history.size()), snap2(history.size());
    CHECK(ds4_gpu_tensor_read(d_residual, 0, actual.data(), actual.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_history, 0, actual_history.data(), actual_history.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_snap, 0, snap.data(), snap.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_snap2, 0, snap2.data(), snap2.size() * sizeof(float)));
    float worst = 0.0f;
    auto compare = [&worst](const std::vector<float> &got, const std::vector<float> &want) {
        for (size_t i = 0; i < got.size(); i++) worst = std::fmax(worst, std::fabs(got[i] - want[i]));
    };
    compare(actual, expected); compare(actual_history, expected_history);
    compare(snap, expected_snap); compare(snap2, expected_snap2);
    CHECK(worst < 3e-6f);
    ds4_gpu_tensor_free(d_snap2); ds4_gpu_tensor_free(d_snap); ds4_gpu_tensor_free(d_history);
    ds4_gpu_tensor_free(d_normed); ds4_gpu_tensor_free(d_gated); ds4_gpu_tensor_free(d_residual);
    printf("Qwen3.8 CUDA PLE dilated conv %s PASS (worst %.9g)\n", use_f16 ? "f16" : "f32", worst);
}

static void check_router_topk(uint32_t experts = 19, uint32_t used = 5) {
    constexpr uint32_t tokens = 3, in_dim = 64;
    std::vector<float> logits((size_t)tokens * experts), input((size_t)tokens * in_dim), gate(in_dim);
    for (size_t i = 0; i < logits.size(); i++) logits[i] = ((int)(i % 13) - 6) * 0.375f;
    logits[7] = logits[3];
    logits[experts + 8] = logits[experts + 2];
    for (size_t i = 0; i < input.size(); i++) input[i] = ((int)(i % 17) - 8) * 0.03125f;
    for (size_t i = 0; i < gate.size(); i++) gate[i] = ((int)(i % 11) - 5) * 0.046875f;
    std::vector<int32_t> expected_selected((size_t)tokens * used);
    std::vector<float> expected_weight((size_t)tokens * used), expected_gate(tokens);
    for (uint32_t t = 0; t < tokens; t++) {
        float max_value = -INFINITY;
        for (uint32_t e = 0; e < experts; e++) max_value = std::fmax(max_value, logits[(size_t)t * experts + e]);
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < used; slot++) {
            int32_t best = -1;
            for (uint32_t e = 0; e < experts; e++) {
                bool seen = false;
                for (uint32_t p = 0; p < slot; p++) seen |= expected_selected[(size_t)t * used + p] == (int32_t)e;
                if (!seen && (best < 0 || logits[(size_t)t * experts + e] > logits[(size_t)t * experts + best])) best = e;
            }
            expected_selected[(size_t)t * used + slot] = best;
            expected_weight[(size_t)t * used + slot] = std::exp(logits[(size_t)t * experts + best] - max_value);
            sum += expected_weight[(size_t)t * used + slot];
        }
        for (uint32_t slot = 0; slot < used; slot++) expected_weight[(size_t)t * used + slot] /= sum;
        for (uint32_t d = 0; d < in_dim; d++) expected_gate[t] += gate[d] * input[(size_t)t * in_dim + d];
    }
    auto alloc_write = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    CHECK(ds4_gpu_set_model_map(gate.data(), gate.size() * sizeof(float)));
    ds4_gpu_tensor *d_logits = alloc_write(logits), *d_input = alloc_write(input);
    ds4_gpu_tensor *d_selected = ds4_gpu_tensor_alloc(expected_selected.size() * sizeof(int32_t));
    ds4_gpu_tensor *d_weight = ds4_gpu_tensor_alloc(expected_weight.size() * sizeof(float));
    ds4_gpu_tensor *d_gate = ds4_gpu_tensor_alloc(expected_gate.size() * sizeof(float));
    CHECK(d_selected && d_weight && d_gate);
    CHECK(ds4_gpu_qwen4_router_topk_tensor(d_selected, d_weight, d_logits, d_input,
                                            gate.data(), gate.size() * sizeof(float), 0, 0, in_dim, d_gate,
                                            tokens, experts, used));
    CHECK(ds4_gpu_synchronize());
    std::vector<int32_t> selected(expected_selected.size());
    std::vector<float> weight(expected_weight.size()), actual_gate(expected_gate.size());
    CHECK(ds4_gpu_tensor_read(d_selected, 0, selected.data(), selected.size() * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_read(d_weight, 0, weight.data(), weight.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_gate, 0, actual_gate.data(), actual_gate.size() * sizeof(float)));
    CHECK(selected == expected_selected);
    float worst = 0.0f;
    for (size_t i = 0; i < weight.size(); i++) worst = std::fmax(worst, std::fabs(weight[i] - expected_weight[i]));
    for (size_t i = 0; i < actual_gate.size(); i++) worst = std::fmax(worst, std::fabs(actual_gate[i] - expected_gate[i]));
    CHECK(worst < 3e-6f);
    ds4_gpu_tensor_free(d_gate); ds4_gpu_tensor_free(d_weight); ds4_gpu_tensor_free(d_selected);
    ds4_gpu_tensor_free(d_input); ds4_gpu_tensor_free(d_logits);
    printf("Qwen3.8 CUDA router top-k/shared gate PASS (worst %.9g)\n", worst);
}

static void check_moe_lists_reduce() {
    constexpr uint32_t tokens = 2, slots = 3, stride = 4, dim = 67, hc = 4;
    std::vector<int32_t> selected = {2, 0, 2, 1, 2, 4};
    std::vector<float> part((size_t)tokens * stride * dim), weights((size_t)tokens * slots);
    std::vector<float> shared_gate = {-0.4f, 0.7f};
    std::vector<float> residual((size_t)tokens * hc * dim);
    std::vector<float> inject((size_t)tokens * hc * DS4_QWEN4_HC_CHUNKS * hc);
    for (size_t i = 0; i < part.size(); i++) part[i] = ((int)(i % 23) - 11) * 0.017578125f;
    for (size_t i = 0; i < weights.size(); i++) weights[i] = 0.1f + (float)(i % slots) * 0.15f;
    for (size_t i = 0; i < residual.size(); i++) residual[i] = ((int)(i % 13) - 6) * 0.03125f;
    for (size_t i = 0; i < inject.size(); i++) inject[i] = ((int)(i % 19) - 9) * 0.00390625f;
    std::vector<float> expected_out((size_t)tokens * dim), expected_residual = residual;
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t d = 0; d < dim; d++) {
        float value = 0.0f;
        for (uint32_t s = 0; s < slots; s++)
            value += weights[(size_t)t * slots + s] * part[((size_t)t * stride + s) * dim + d];
        value += sigmoid(shared_gate[t]) * part[((size_t)t * stride + slots) * dim + d];
        expected_out[(size_t)t * dim + d] = value;
        for (uint32_t stream = 0; stream < hc; stream++) {
            float sum = 0.0f;
            const size_t ib = (size_t)t * hc * DS4_QWEN4_HC_CHUNKS * hc;
            for (uint32_t source = 0; source < hc * DS4_QWEN4_HC_CHUNKS; source++)
                sum += inject[ib + (size_t)source * hc + stream];
            expected_residual[((size_t)t * hc + stream) * dim + d] += 2.0f * sigmoid(sum / hc) * value;
        }
    }
    auto alloc_float = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    ds4_gpu_tensor *d_part = alloc_float(part), *d_weights = alloc_float(weights);
    ds4_gpu_tensor *d_gate = alloc_float(shared_gate), *d_residual = alloc_float(residual), *d_inject = alloc_float(inject);
    ds4_gpu_tensor *d_out = ds4_gpu_tensor_alloc(expected_out.size() * sizeof(float));
    ds4_gpu_tensor *d_selected = ds4_gpu_tensor_alloc(selected.size() * sizeof(int32_t));
    CHECK(d_out && d_selected && ds4_gpu_tensor_write(d_selected, 0, selected.data(), selected.size() * sizeof(int32_t)));
    CHECK(ds4_gpu_qwen4_moe_reduce_tensor(d_out, d_part, d_weights, d_gate, NULL, d_residual, d_inject,
                                           tokens, slots, stride, dim, hc));
    constexpr uint32_t experts = 5, cap = 2;
    ds4_gpu_tensor *d_lists = ds4_gpu_tensor_alloc((size_t)experts * cap * sizeof(int32_t));
    ds4_gpu_tensor *d_counts = ds4_gpu_tensor_alloc(experts * sizeof(int32_t));
    CHECK(d_lists && d_counts);
    CHECK(ds4_gpu_qwen4_moe_build_lists_tensor(d_lists, d_counts, d_selected, tokens, slots, experts, cap));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual_out(expected_out.size()), actual_residual(residual.size());
    std::vector<int32_t> lists((size_t)experts * cap), counts(experts);
    CHECK(ds4_gpu_tensor_read(d_out, 0, actual_out.data(), actual_out.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_residual, 0, actual_residual.data(), actual_residual.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_lists, 0, lists.data(), lists.size() * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_read(d_counts, 0, counts.data(), counts.size() * sizeof(int32_t)));
    const std::vector<int32_t> expected_counts = {1, 1, 2, 0, 1};
    CHECK(counts == expected_counts);
    CHECK(lists[0] == 1 && lists[2] == 3 && lists[4] == 0 && lists[5] == 2 && lists[8] == 5);
    float worst = 0.0f;
    for (size_t i = 0; i < actual_out.size(); i++) worst = std::fmax(worst, std::fabs(actual_out[i] - expected_out[i]));
    for (size_t i = 0; i < actual_residual.size(); i++) worst = std::fmax(worst, std::fabs(actual_residual[i] - expected_residual[i]));
    CHECK(worst < 3e-6f);
    ds4_gpu_tensor_free(d_counts); ds4_gpu_tensor_free(d_lists); ds4_gpu_tensor_free(d_selected);
    ds4_gpu_tensor_free(d_out); ds4_gpu_tensor_free(d_inject); ds4_gpu_tensor_free(d_residual);
    ds4_gpu_tensor_free(d_gate); ds4_gpu_tensor_free(d_weights); ds4_gpu_tensor_free(d_part);
    printf("Qwen3.8 CUDA MoE lists/reduce/HC combine PASS (worst %.9g)\n", worst);
}

static void rope_neox_ref(float *row, uint32_t n_rot, const uint32_t *pos3,
                          const float *freq, float mscale) {
    const uint32_t half = n_rot / 2u;
    for (uint32_t i = 0; i < half; i++) {
        const float theta = (float)pos3[i % 3u] * freq[i];
        const float c = std::cos(theta) * mscale, s = std::sin(theta) * mscale;
        const float x0 = row[i], x1 = row[i + half];
        row[i] = x0 * c - x1 * s;
        row[i + half] = x0 * s + x1 * c;
    }
}

static void check_attn_prep() {
    constexpr uint32_t tokens = 2, heads = 2, kv_heads = 1, dim = 64;
    constexpr uint32_t idx_heads = 2, idx_dim = 64, n_rot = 32, pos0 = 2, cap = 6;
    std::vector<float> model((size_t)2 * dim + idx_dim), freq(32);
    for (uint32_t d = 0; d < dim; d++) {
        model[d] = 0.8f + (d % 7) * 0.03125f;
        model[dim + d] = 0.7f + (d % 9) * 0.02734375f;
    }
    for (uint32_t d = 0; d < idx_dim; d++) model[2u * dim + d] = 0.9f + (d % 5) * 0.0234375f;
    for (uint32_t i = 0; i < 32; i++) freq[i] = 0.01f * (float)(i + 1u);
    constexpr float mscale = 0.875f, eps = 1e-6f;
    std::vector<float> qg((size_t)tokens * heads * 2u * dim);
    std::vector<float> kp((size_t)tokens * kv_heads * dim), vp(kp.size());
    std::vector<float> iq((size_t)tokens * idx_heads * idx_dim), ik((size_t)tokens * idx_dim);
    std::vector<uint32_t> positions((size_t)cap * 4u);
    for (size_t i = 0; i < qg.size(); i++) qg[i] = ((int)(i % 31) - 15) * 0.01953125f;
    for (size_t i = 0; i < kp.size(); i++) { kp[i] = ((int)(i % 23) - 11) * 0.02734375f; vp[i] = ((int)(i % 17) - 8) * 0.0390625f; }
    for (size_t i = 0; i < iq.size(); i++) iq[i] = ((int)(i % 29) - 14) * 0.015625f;
    for (size_t i = 0; i < ik.size(); i++) ik[i] = ((int)(i % 13) - 6) * 0.046875f;
    for (uint32_t p = 0; p < cap; p++) {
        positions[(size_t)p * 4u] = p + 1u;
        positions[(size_t)p * 4u + 1u] = 2u * p + 1u;
        positions[(size_t)p * 4u + 2u] = 3u * p + 2u;
    }
    std::vector<float> expected_q((size_t)tokens * heads * dim), expected_gate(expected_q.size());
    std::vector<float> expected_iq(iq.size()), expected_ik((size_t)cap * idx_dim, 0.0f);
    std::vector<__half> expected_k((size_t)cap * kv_heads * dim), expected_v(expected_k.size());
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t *p3 = positions.data() + (size_t)(pos0 + t) * 4u;
        for (uint32_t h = 0; h < heads; h++) {
            const float *src = qg.data() + ((size_t)t * heads + h) * 2u * dim;
            float *dst = expected_q.data() + ((size_t)t * heads + h) * dim;
            float sum = 0.0f;
            for (uint32_t d = 0; d < dim; d++) sum += src[d] * src[d];
            const float scale = 1.0f / std::sqrt(sum / dim + eps);
            for (uint32_t d = 0; d < dim; d++) { dst[d] = src[d] * scale * model[d]; expected_gate[((size_t)t * heads + h) * dim + d] = src[dim + d]; }
            rope_neox_ref(dst, n_rot, p3, freq.data(), mscale);
        }
        {
            const float *src = kp.data() + (size_t)t * dim;
            float row[dim], sum = 0.0f;
            for (uint32_t d = 0; d < dim; d++) sum += src[d] * src[d];
            const float scale = 1.0f / std::sqrt(sum / dim + eps);
            for (uint32_t d = 0; d < dim; d++) row[d] = src[d] * scale * model[dim + d];
            rope_neox_ref(row, n_rot, p3, freq.data(), mscale);
            for (uint32_t d = 0; d < dim; d++) {
                expected_k[((size_t)(pos0 + t) * kv_heads) * dim + d] = __float2half(row[d]);
                expected_v[((size_t)(pos0 + t) * kv_heads) * dim + d] = __float2half(vp[(size_t)t * dim + d]);
            }
        }
        for (uint32_t h = 0; h < idx_heads; h++) {
            const float *src = iq.data() + ((size_t)t * idx_heads + h) * idx_dim;
            float *dst = expected_iq.data() + ((size_t)t * idx_heads + h) * idx_dim;
            float sum = 0.0f;
            for (uint32_t d = 0; d < idx_dim; d++) sum += src[d] * src[d];
            const float scale = 1.0f / std::sqrt(sum / idx_dim + eps);
            for (uint32_t d = 0; d < idx_dim; d++) dst[d] = src[d] * scale * model[2u * dim + d];
            rope_neox_ref(dst, n_rot, p3, freq.data(), mscale);
        }
        for (uint32_t d = 0; d < idx_dim; d++) expected_ik[(size_t)(pos0 + t) * idx_dim + d] = ik[(size_t)t * idx_dim + d];
    }
    auto alloc_float = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    CHECK(ds4_gpu_set_model_map(model.data(), model.size() * sizeof(float)));
    ds4_gpu_tensor *d_qg = alloc_float(qg), *d_kp = alloc_float(kp), *d_vp = alloc_float(vp);
    ds4_gpu_tensor *d_iq = alloc_float(iq), *d_ik = alloc_float(ik);
    ds4_gpu_tensor *d_pos = ds4_gpu_tensor_alloc(positions.size() * sizeof(uint32_t));
    CHECK(d_pos && ds4_gpu_tensor_write(d_pos, 0, positions.data(), positions.size() * sizeof(uint32_t)));
    ds4_gpu_tensor *d_q = ds4_gpu_tensor_alloc(expected_q.size() * sizeof(float));
    ds4_gpu_tensor *d_gate = ds4_gpu_tensor_alloc(expected_gate.size() * sizeof(float));
    ds4_gpu_tensor *d_kcache = ds4_gpu_tensor_alloc(expected_k.size() * sizeof(__half));
    ds4_gpu_tensor *d_vcache = ds4_gpu_tensor_alloc(expected_v.size() * sizeof(__half));
    ds4_gpu_tensor *d_iqout = ds4_gpu_tensor_alloc(expected_iq.size() * sizeof(float));
    ds4_gpu_tensor *d_ikcache = ds4_gpu_tensor_alloc(expected_ik.size() * sizeof(float));
    CHECK(d_q && d_gate && d_kcache && d_vcache && d_iqout && d_ikcache);
    ds4_gpu_qwen4_set_rope(freq.data(), 32, mscale);
    CHECK(ds4_gpu_qwen4_attn_prep_tensor(d_q, d_gate, d_kcache, d_vcache, d_iqout, d_ikcache,
                                          d_qg, d_kp, d_vp, d_iq, d_ik, d_pos,
                                          model.data(), model.size() * sizeof(float), 0, dim * sizeof(float),
                                          2u * dim * sizeof(float), tokens, heads, kv_heads, dim, n_rot,
                                          idx_heads, idx_dim, pos0, cap, 10000.0f, eps));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual_q(expected_q.size()), actual_gate(expected_gate.size()), actual_iq(expected_iq.size()), actual_ik(expected_ik.size());
    std::vector<__half> actual_k(expected_k.size()), actual_v(expected_v.size());
    CHECK(ds4_gpu_tensor_read(d_q, 0, actual_q.data(), actual_q.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_gate, 0, actual_gate.data(), actual_gate.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_iqout, 0, actual_iq.data(), actual_iq.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_ikcache, 0, actual_ik.data(), actual_ik.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_kcache, 0, actual_k.data(), actual_k.size() * sizeof(__half)));
    CHECK(ds4_gpu_tensor_read(d_vcache, 0, actual_v.data(), actual_v.size() * sizeof(__half)));
    float worst = 0.0f;
    auto compare = [&worst](const std::vector<float> &got, const std::vector<float> &want) {
        for (size_t i = 0; i < got.size(); i++) worst = std::fmax(worst, std::fabs(got[i] - want[i]));
    };
    compare(actual_q, expected_q); compare(actual_gate, expected_gate); compare(actual_iq, expected_iq);
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t d = 0; d < idx_dim; d++)
        worst = std::fmax(worst, std::fabs(actual_ik[(size_t)(pos0 + t) * idx_dim + d] - expected_ik[(size_t)(pos0 + t) * idx_dim + d]));
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t d = 0; d < dim; d++) {
        const size_t i = (size_t)(pos0 + t) * dim + d;
        worst = std::fmax(worst, std::fabs(__half2float(actual_k[i]) - __half2float(expected_k[i])));
        worst = std::fmax(worst, std::fabs(__half2float(actual_v[i]) - __half2float(expected_v[i])));
    }
    CHECK(worst < 5e-6f);
    ds4_gpu_qwen4_set_rope(NULL, 0, 1.0f);
    ds4_gpu_tensor_free(d_ikcache); ds4_gpu_tensor_free(d_iqout); ds4_gpu_tensor_free(d_vcache); ds4_gpu_tensor_free(d_kcache);
    ds4_gpu_tensor_free(d_gate); ds4_gpu_tensor_free(d_q); ds4_gpu_tensor_free(d_pos); ds4_gpu_tensor_free(d_ik);
    ds4_gpu_tensor_free(d_iq); ds4_gpu_tensor_free(d_vp); ds4_gpu_tensor_free(d_kp); ds4_gpu_tensor_free(d_qg);
    printf("Qwen3.8 CUDA attention prep/RoPE/KV cache PASS (worst %.9g)\n", worst);
}

static void check_qsa_indexer() {
    constexpr uint32_t ratio = 4, blocks = 3, idx_dim = 64, n_rot = 32;
    constexpr uint32_t tokens = 2, idx_heads = 2, pos0 = 10, top_k = 2, stride = 11;
    std::vector<float> gamma(idx_dim), ik_cache((size_t)blocks * ratio * idx_dim);
    std::vector<uint32_t> positions((size_t)blocks * ratio * 4u);
    for (uint32_t d = 0; d < idx_dim; d++) gamma[d] = 0.8f + (d % 7) * 0.03125f;
    for (size_t i = 0; i < ik_cache.size(); i++) ik_cache[i] = ((int)(i % 29) - 14) * 0.01953125f;
    for (uint32_t p = 0; p < blocks * ratio; p++) {
        positions[(size_t)p * 4u] = p;
        positions[(size_t)p * 4u + 1u] = p + 2u;
        positions[(size_t)p * 4u + 2u] = 2u * p + 1u;
    }
    std::vector<float> freq(32);
    for (uint32_t i = 0; i < n_rot / 2u; i++) freq[i] = std::pow(10000.0f, -2.0f * i / n_rot);
    std::vector<__half> expected_key((size_t)blocks * idx_dim);
    constexpr float eps = 1e-6f;
    for (uint32_t b = 0; b < blocks; b++) {
        float row[idx_dim], sum = 0.0f;
        for (uint32_t d = 0; d < idx_dim; d++) {
            row[d] = 0.0f;
            for (uint32_t t = 0; t < ratio; t++) row[d] += ik_cache[((size_t)b * ratio + t) * idx_dim + d];
            row[d] /= ratio;
            sum += row[d] * row[d];
        }
        const float scale = 1.0f / std::sqrt(sum / idx_dim + eps);
        for (uint32_t d = 0; d < idx_dim; d++) row[d] *= scale * gamma[d];
        rope_neox_ref(row, n_rot, positions.data() + (size_t)b * ratio * 4u, freq.data(), 1.0f);
        for (uint32_t d = 0; d < idx_dim; d++) expected_key[(size_t)b * idx_dim + d] = __float2half(row[d]);
    }
    std::vector<float> iq((size_t)tokens * idx_heads * idx_dim);
    for (size_t i = 0; i < iq.size(); i++) iq[i] = ((int)(i % 23) - 11) * 0.02734375f;
    std::vector<float> expected_score((size_t)tokens * blocks);
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t b = 0; b < blocks; b++) {
        if (b >= (pos0 + t + 1u) / ratio) { expected_score[(size_t)t * blocks + b] = -3.0e38f; continue; }
        float score = 0.0f;
        for (uint32_t h = 0; h < idx_heads; h++) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < idx_dim; d++)
                dot += iq[((size_t)t * idx_heads + h) * idx_dim + d] * __half2float(expected_key[(size_t)b * idx_dim + d]);
            score += std::fmax(dot, 0.0f);
        }
        expected_score[(size_t)t * blocks + b] = score;
    }
    std::vector<int32_t> expected_sel((size_t)tokens * top_k);
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t k = 0; k < top_k; k++) {
        int32_t best = -1;
        for (uint32_t b = 0; b < blocks; b++) {
            bool used = false;
            for (uint32_t p = 0; p < k; p++) used |= expected_sel[(size_t)t * top_k + p] == (int32_t)b;
            if (!used && (best < 0 || expected_score[(size_t)t * blocks + b] > expected_score[(size_t)t * blocks + best])) best = b;
        }
        expected_sel[(size_t)t * top_k + k] = best;
    }
    auto alloc_float = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    CHECK(ds4_gpu_set_model_map(gamma.data(), gamma.size() * sizeof(float)));
    ds4_gpu_tensor *d_ik = alloc_float(ik_cache), *d_iq = alloc_float(iq);
    ds4_gpu_tensor *d_pos = ds4_gpu_tensor_alloc(positions.size() * sizeof(uint32_t));
    ds4_gpu_tensor *d_key = ds4_gpu_tensor_alloc(expected_key.size() * sizeof(__half));
    ds4_gpu_tensor *d_score = ds4_gpu_tensor_alloc(expected_score.size() * sizeof(float));
    ds4_gpu_tensor *d_tile = ds4_gpu_tensor_alloc(tokens * sizeof(uint32_t));
    ds4_gpu_tensor *d_sel = ds4_gpu_tensor_alloc(expected_sel.size() * sizeof(int32_t));
    ds4_gpu_tensor *d_tokens = ds4_gpu_tensor_alloc((size_t)tokens * stride * sizeof(int32_t));
    ds4_gpu_tensor *d_count = ds4_gpu_tensor_alloc(tokens * sizeof(uint32_t));
    CHECK(d_pos && d_key && d_score && d_tile && d_sel && d_tokens && d_count);
    CHECK(ds4_gpu_tensor_write(d_pos, 0, positions.data(), positions.size() * sizeof(uint32_t)));
    CHECK(ds4_gpu_qwen4_idx_block_key_tensor(d_key, d_ik, d_pos, gamma.data(), gamma.size() * sizeof(float),
                                               0, 0, blocks, ratio, idx_dim, n_rot, 10000.0f, eps));
    CHECK(ds4_gpu_qwen4_idx_score_tensor(d_score, d_tile, d_iq, d_key,
                                          tokens, blocks, idx_heads, idx_dim, pos0, ratio));
    CHECK(ds4_gpu_qwen4_idx_select_tensor(d_sel, d_score, d_tile, blocks, tokens, top_k));
    CHECK(ds4_gpu_qwen4_idx_expand_tensor(d_tokens, d_count, d_sel, tokens, top_k, ratio, pos0, stride));
    CHECK(ds4_gpu_synchronize());
    std::vector<__half> actual_key(expected_key.size());
    std::vector<float> actual_score(expected_score.size());
    std::vector<int32_t> actual_sel(expected_sel.size()), actual_tokens((size_t)tokens * stride);
    std::vector<uint32_t> counts(tokens);
    CHECK(ds4_gpu_tensor_read(d_key, 0, actual_key.data(), actual_key.size() * sizeof(__half)));
    CHECK(ds4_gpu_tensor_read(d_score, 0, actual_score.data(), actual_score.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_sel, 0, actual_sel.data(), actual_sel.size() * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_read(d_tokens, 0, actual_tokens.data(), actual_tokens.size() * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_read(d_count, 0, counts.data(), counts.size() * sizeof(uint32_t)));
    CHECK(actual_sel == expected_sel);
    CHECK(counts[0] == 11u && counts[1] == 8u);
    float worst = 0.0f;
    for (size_t i = 0; i < actual_key.size(); i++) worst = std::fmax(worst, std::fabs(__half2float(actual_key[i]) - __half2float(expected_key[i])));
    for (size_t i = 0; i < actual_score.size(); i++) worst = std::fmax(worst, std::fabs(actual_score[i] - expected_score[i]));
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t i = 0; i < top_k * ratio; i++)
            CHECK(actual_tokens[(size_t)t * stride + i] == expected_sel[(size_t)t * top_k + i / ratio] * (int32_t)ratio + (int32_t)(i % ratio));
        const uint32_t tail_start = ((pos0 + t + 1u) / ratio) * ratio;
        for (uint32_t p = tail_start; p <= pos0 + t; p++)
            CHECK(actual_tokens[(size_t)t * stride + top_k * ratio + p - tail_start] == (int32_t)p);
    }
    CHECK(worst < 8e-6f);
    ds4_gpu_tensor_free(d_count); ds4_gpu_tensor_free(d_tokens); ds4_gpu_tensor_free(d_sel); ds4_gpu_tensor_free(d_tile);
    ds4_gpu_tensor_free(d_score); ds4_gpu_tensor_free(d_key); ds4_gpu_tensor_free(d_pos); ds4_gpu_tensor_free(d_iq); ds4_gpu_tensor_free(d_ik);
    printf("Qwen3.8 CUDA QSA block/score/select/expand PASS (worst %.9g)\n", worst);
}

static void check_attention_decode() {
    constexpr uint32_t tokens = 2, heads = 4, kv_heads = 2, dim = 32, pos0 = 3, cap = 5;
    constexpr uint32_t stride = 3;
    std::vector<float> query((size_t)tokens * heads * dim), gate(query.size());
    std::vector<__half> key((size_t)cap * kv_heads * dim), value(key.size());
    for (size_t i = 0; i < query.size(); i++) { query[i] = ((int)(i % 23) - 11) * 0.03125f; gate[i] = ((int)(i % 17) - 8) * 0.078125f; }
    for (size_t i = 0; i < key.size(); i++) { key[i] = __float2half(((int)(i % 29) - 14) * 0.0234375f); value[i] = __float2half(((int)(i % 19) - 9) * 0.03515625f); }
    std::vector<int32_t> selected = {0, 3, -1, 1, 2, 4};
    std::vector<uint32_t> counts = {2, 3};
    const float attn_scale = 1.0f / std::sqrt((float)dim);
    auto reference = [&](bool sparse) {
        std::vector<float> out(query.size());
        for (uint32_t t = 0; t < tokens; t++) for (uint32_t h = 0; h < heads; h++) {
            const uint32_t kvh = h / (heads / kv_heads);
            const float *q = query.data() + ((size_t)t * heads + h) * dim;
            float maximum = -INFINITY, denominator = 0.0f, acc[dim] = {0};
            const uint32_t n = sparse ? counts[t] : pos0 + t + 1u;
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t p = sparse ? (uint32_t)selected[(size_t)t * stride + i] : i;
                float score = 0.0f;
                for (uint32_t d = 0; d < dim; d++) score += q[d] * __half2float(key[((size_t)p * kv_heads + kvh) * dim + d]);
                score *= attn_scale;
                const float next = std::fmax(maximum, score);
                const float old_scale = denominator == 0.0f ? 0.0f : std::exp(maximum - next);
                const float new_scale = std::exp(score - next);
                denominator = denominator * old_scale + new_scale;
                for (uint32_t d = 0; d < dim; d++)
                    acc[d] = acc[d] * old_scale + new_scale * __half2float(value[((size_t)p * kv_heads + kvh) * dim + d]);
                maximum = next;
            }
            for (uint32_t d = 0; d < dim; d++) {
                const size_t o = ((size_t)t * heads + h) * dim + d;
                out[o] = acc[d] / denominator * sigmoid(gate[o]);
            }
        }
        return out;
    };
    const std::vector<float> expected_dense = reference(false), expected_sparse = reference(true);
    auto alloc_float = [](const std::vector<float> &host) {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(host.size() * sizeof(float));
        CHECK(tensor && ds4_gpu_tensor_write(tensor, 0, host.data(), host.size() * sizeof(float)));
        return tensor;
    };
    ds4_gpu_tensor *d_q = alloc_float(query), *d_gate = alloc_float(gate);
    ds4_gpu_tensor *d_key = ds4_gpu_tensor_alloc(key.size() * sizeof(__half));
    ds4_gpu_tensor *d_value = ds4_gpu_tensor_alloc(value.size() * sizeof(__half));
    ds4_gpu_tensor *d_sel = ds4_gpu_tensor_alloc(selected.size() * sizeof(int32_t));
    ds4_gpu_tensor *d_count = ds4_gpu_tensor_alloc(counts.size() * sizeof(uint32_t));
    ds4_gpu_tensor *d_dense = ds4_gpu_tensor_alloc(query.size() * sizeof(float));
    ds4_gpu_tensor *d_sparse = ds4_gpu_tensor_alloc(query.size() * sizeof(float));
    CHECK(d_key && d_value && d_sel && d_count && d_dense && d_sparse);
    CHECK(ds4_gpu_tensor_write(d_key, 0, key.data(), key.size() * sizeof(__half)));
    CHECK(ds4_gpu_tensor_write(d_value, 0, value.data(), value.size() * sizeof(__half)));
    CHECK(ds4_gpu_tensor_write(d_sel, 0, selected.data(), selected.size() * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(d_count, 0, counts.data(), counts.size() * sizeof(uint32_t)));
    CHECK(ds4_gpu_qwen4_attn_decode_tensor(d_dense, d_q, d_gate, d_key, d_value, NULL, NULL, NULL,
                                            tokens, heads, kv_heads, dim, pos0, false, 0, attn_scale));
    CHECK(ds4_gpu_qwen4_attn_decode_tensor(d_sparse, d_q, d_gate, d_key, d_value, d_sel, d_count, NULL,
                                            tokens, heads, kv_heads, dim, pos0, true, stride, attn_scale));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual_dense(query.size()), actual_sparse(query.size());
    CHECK(ds4_gpu_tensor_read(d_dense, 0, actual_dense.data(), actual_dense.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_sparse, 0, actual_sparse.data(), actual_sparse.size() * sizeof(float)));
    float worst = 0.0f;
    for (size_t i = 0; i < query.size(); i++) {
        worst = std::fmax(worst, std::fabs(actual_dense[i] - expected_dense[i]));
        worst = std::fmax(worst, std::fabs(actual_sparse[i] - expected_sparse[i]));
    }
    CHECK(worst < 5e-6f);
    ds4_gpu_tensor_free(d_sparse); ds4_gpu_tensor_free(d_dense); ds4_gpu_tensor_free(d_count); ds4_gpu_tensor_free(d_sel);
    ds4_gpu_tensor_free(d_value); ds4_gpu_tensor_free(d_key); ds4_gpu_tensor_free(d_gate); ds4_gpu_tensor_free(d_q);
    printf("Qwen3.8 CUDA dense/sparse attention decode PASS (worst %.9g)\n", worst);
}

static void make_q4k_block(uint8_t *block, uint32_t seed) {
    memset(block, 0, 144);
    const __half d = __float2half(0.00390625f * (1u + seed % 5u));
    const __half dmin = __float2half(0.001953125f * (1u + seed % 3u));
    memcpy(block, &d, sizeof(d));
    memcpy(block + 2, &dmin, sizeof(dmin));
    for (uint32_t g = 0; g < 4; g++) {
        block[4 + g] = (uint8_t)(1u + (seed + g) % 11u);
        block[8 + g] = (uint8_t)(1u + (seed + 2u * g) % 7u);
    }
    for (uint32_t g = 4; g < 8; g++) {
        const uint8_t scale = (uint8_t)(1u + (seed + g) % 13u);
        const uint8_t minimum = (uint8_t)(1u + (seed + 3u * g) % 9u);
        block[12 + g - 4u] = (uint8_t)(scale | (minimum << 4u));
    }
    for (uint32_t i = 0; i < 128; i++) block[16 + i] = (uint8_t)((seed * 17u + i * 29u) & 255u);
}

static float q4k_value_ref(const uint8_t *block, uint32_t index) {
    __half hd, hm;
    memcpy(&hd, block, sizeof(hd)); memcpy(&hm, block + 2, sizeof(hm));
    const uint32_t group = index / 32u, within = index % 32u;
    const uint8_t *sc = block + 4;
    uint32_t scale, minimum;
    if (group < 4u) { scale = sc[group] & 63u; minimum = sc[group + 4u] & 63u; }
    else {
        scale = (sc[group + 4u] & 15u) | ((sc[group - 4u] & 0xc0u) >> 2u);
        minimum = (sc[group + 4u] >> 4u) | ((sc[group] & 0xc0u) >> 2u);
    }
    const uint8_t packed = block[16u + (group >> 1u) * 32u + within];
    const uint32_t q = (packed >> ((group & 1u) * 4u)) & 15u;
    return __half2float(hd) * scale * q - __half2float(hm) * minimum;
}

static void check_moe_q4k_mid_down() {
    constexpr uint32_t experts = 2, tokens = 1, slots = 2, in_dim = 256, ff_dim = 256, out_dim = 5;
    constexpr uint64_t row_bytes = 144, mid_expert_bytes = row_bytes * ff_dim;
    constexpr uint64_t down_expert_bytes = row_bytes * out_dim;
    const uint64_t gate_off = 0;
    const uint64_t up_off = gate_off + experts * mid_expert_bytes;
    const uint64_t shared_gate_off = up_off + experts * mid_expert_bytes;
    const uint64_t shared_up_off = shared_gate_off + mid_expert_bytes;
    const uint64_t down_off = shared_up_off + mid_expert_bytes;
    const uint64_t shared_down_off = down_off + experts * down_expert_bytes;
    std::vector<uint8_t> model(shared_down_off + down_expert_bytes);
    uint32_t seed = 1;
    auto fill_rows = [&](uint64_t offset, uint32_t rows) {
        for (uint32_t r = 0; r < rows; r++) make_q4k_block(model.data() + offset + (uint64_t)r * row_bytes, seed++);
    };
    fill_rows(gate_off, experts * ff_dim); fill_rows(up_off, experts * ff_dim);
    fill_rows(shared_gate_off, ff_dim); fill_rows(shared_up_off, ff_dim);
    fill_rows(down_off, experts * out_dim); fill_rows(shared_down_off, out_dim);
    std::vector<float> input(in_dim);
    for (uint32_t i = 0; i < in_dim; i++) input[i] = ((int)(i % 31) - 15) * 0.015625f;
    std::vector<int32_t> selected = {1, 0};
    constexpr uint32_t n_out = slots + 1u;
    std::vector<float> expected_mid((size_t)n_out * ff_dim), expected_part((size_t)n_out * out_dim);
    auto dot = [&](uint64_t offset, const float *x) {
        float sum = 0.0f;
        const uint8_t *row = model.data() + offset;
        for (uint32_t i = 0; i < 256; i++) sum += q4k_value_ref(row, i) * x[i];
        return sum;
    };
    for (uint32_t slot = 0; slot < n_out; slot++) {
        const bool shared = slot == slots;
        const uint32_t expert = shared ? 0u : (uint32_t)selected[slot];
        const uint64_t gb = shared ? shared_gate_off : gate_off + (uint64_t)expert * mid_expert_bytes;
        const uint64_t ub = shared ? shared_up_off : up_off + (uint64_t)expert * mid_expert_bytes;
        for (uint32_t r = 0; r < ff_dim; r++) {
            const float g = dot(gb + (uint64_t)r * row_bytes, input.data());
            const float u = dot(ub + (uint64_t)r * row_bytes, input.data());
            expected_mid[(size_t)slot * ff_dim + r] = g * sigmoid(g) * u;
        }
        const uint64_t db = shared ? shared_down_off : down_off + (uint64_t)expert * down_expert_bytes;
        for (uint32_t r = 0; r < out_dim; r++)
            expected_part[(size_t)slot * out_dim + r] = dot(db + (uint64_t)r * row_bytes,
                                                            expected_mid.data() + (size_t)slot * ff_dim);
    }
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    ds4_gpu_tensor *d_input = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    ds4_gpu_tensor *d_selected = ds4_gpu_tensor_alloc(selected.size() * sizeof(int32_t));
    ds4_gpu_tensor *d_mid = ds4_gpu_tensor_alloc(expected_mid.size() * sizeof(float));
    ds4_gpu_tensor *d_part = ds4_gpu_tensor_alloc(expected_part.size() * sizeof(float));
    CHECK(d_input && d_selected && d_mid && d_part);
    CHECK(ds4_gpu_tensor_write(d_input, 0, input.data(), input.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(d_selected, 0, selected.data(), selected.size() * sizeof(int32_t)));
    CHECK(ds4_gpu_qwen4_moe_mid_tensor(d_mid, d_input, d_selected, model.data(), model.size(), gate_off, up_off,
                                        12, experts, tokens, slots, in_dim, ff_dim,
                                        shared_gate_off, shared_up_off, 12));
    CHECK(ds4_gpu_qwen4_moe_down_tensor(d_part, d_mid, d_selected, model.data(), model.size(), down_off,
                                         12, experts, tokens, slots, ff_dim, out_dim, shared_down_off, 12));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual_mid(expected_mid.size()), actual_part(expected_part.size());
    CHECK(ds4_gpu_tensor_read(d_mid, 0, actual_mid.data(), actual_mid.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_part, 0, actual_part.data(), actual_part.size() * sizeof(float)));
    float worst_mid = 0.0f, worst_part = 0.0f;
    for (size_t i = 0; i < actual_mid.size(); i++) worst_mid = std::fmax(worst_mid, std::fabs(actual_mid[i] - expected_mid[i]));
    for (size_t i = 0; i < actual_part.size(); i++) worst_part = std::fmax(worst_part, std::fabs(actual_part[i] - expected_part[i]));
    CHECK(worst_mid < 2e-4f && worst_part < 2e-3f);

    ds4_gpu_tensor *d_lists = ds4_gpu_tensor_alloc(experts * sizeof(int32_t));
    ds4_gpu_tensor *d_counts = ds4_gpu_tensor_alloc(experts * sizeof(int32_t));
    ds4_gpu_tensor *d_mm_mid = ds4_gpu_tensor_alloc((size_t)slots * ff_dim * sizeof(float));
    ds4_gpu_tensor *d_mm_part = ds4_gpu_tensor_alloc((size_t)slots * out_dim * sizeof(float));
    CHECK(d_lists && d_counts && d_mm_mid && d_mm_part);
    CHECK(ds4_gpu_qwen4_moe_build_lists_tensor(d_lists, d_counts, d_selected, tokens, slots, experts, 1));
    CHECK(ds4_gpu_qwen4_moe_mm_mid_tensor(d_mm_mid, d_input, d_lists, d_counts, model.data(), model.size(),
                                           gate_off, up_off, 12, experts, tokens, slots, slots,
                                           in_dim, ff_dim, 1));
    CHECK(ds4_gpu_qwen4_moe_mm_down_tensor(d_mm_part, d_mm_mid, d_lists, d_counts, model.data(), model.size(),
                                            down_off, 12, experts, tokens, slots, slots,
                                            ff_dim, out_dim, 1));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> mm_mid((size_t)slots * ff_dim), mm_part((size_t)slots * out_dim);
    CHECK(ds4_gpu_tensor_read(d_mm_mid, 0, mm_mid.data(), mm_mid.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(d_mm_part, 0, mm_part.data(), mm_part.size() * sizeof(float)));
    for (size_t i = 0; i < mm_mid.size(); i++) worst_mid = std::fmax(worst_mid, std::fabs(mm_mid[i] - expected_mid[i]));
    for (size_t i = 0; i < mm_part.size(); i++) worst_part = std::fmax(worst_part, std::fabs(mm_part[i] - expected_part[i]));
    CHECK(worst_mid < 2e-4f && worst_part < 2e-3f);
    // Exercise multiple grid-stride iterations and unused list capacity.
    {
        constexpr uint32_t batch = 9, capacity = 16;
        std::vector<float> bx(batch * in_dim);
        std::vector<int32_t> bl(experts * capacity, -1), bc(experts, batch);
        std::vector<float> source(in_dim);
        CHECK(ds4_gpu_tensor_read(d_input, 0, source.data(), in_dim * sizeof(float)));
        for (uint32_t t = 0; t < batch; ++t) {
            for (uint32_t i = 0; i < in_dim; ++i) bx[t * in_dim + i] = source[i];
            for (uint32_t s = 0; s < slots; ++s) bl[selected[s] * capacity + t] = t * slots + s;
        }
        auto x = ds4_gpu_tensor_alloc(bx.size() * sizeof(float));
        auto lists = ds4_gpu_tensor_alloc(bl.size() * sizeof(int32_t));
        auto counts = ds4_gpu_tensor_alloc(bc.size() * sizeof(int32_t));
        auto mid = ds4_gpu_tensor_alloc(batch * slots * ff_dim * sizeof(float));
        auto part = ds4_gpu_tensor_alloc(batch * slots * out_dim * sizeof(float));
        CHECK(x && lists && counts && mid && part);
        CHECK(ds4_gpu_tensor_write(x, 0, bx.data(), bx.size() * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(lists, 0, bl.data(), bl.size() * sizeof(int32_t)));
        CHECK(ds4_gpu_tensor_write(counts, 0, bc.data(), bc.size() * sizeof(int32_t)));
        CHECK(ds4_gpu_qwen4_moe_mm_mid_tensor(mid, x, lists, counts, model.data(), model.size(),
            gate_off, up_off, 12, experts, batch, slots, slots, in_dim, ff_dim, capacity));
        CHECK(ds4_gpu_qwen4_moe_mm_down_tensor(part, mid, lists, counts, model.data(), model.size(),
            down_off, 12, experts, batch, slots, slots, ff_dim, out_dim, capacity));
        std::vector<float> result(batch * slots * out_dim);
        CHECK(ds4_gpu_tensor_read(part, 0, result.data(), result.size() * sizeof(float)));
        for (size_t i = 0; i < result.size(); ++i)
            CHECK(std::fabs(result[i] - mm_part[i % mm_part.size()]) < 2e-3f);
        ds4_gpu_tensor_free(part); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(counts);
        ds4_gpu_tensor_free(lists); ds4_gpu_tensor_free(x);
    }
    ds4_gpu_tensor_free(d_mm_part); ds4_gpu_tensor_free(d_mm_mid);
    ds4_gpu_tensor_free(d_counts); ds4_gpu_tensor_free(d_lists);
    ds4_gpu_tensor_free(d_part); ds4_gpu_tensor_free(d_mid); ds4_gpu_tensor_free(d_selected); ds4_gpu_tensor_free(d_input);
    printf("Qwen3.8 CUDA Q4_K MoE routed/list/shared PASS (worst mid %.9g down %.9g)\n", worst_mid, worst_part);
}

static void check_dense_q8_float_input(uint32_t dim, uint32_t rows) {
    constexpr uint32_t max_tokens = 33;
    std::vector<uint8_t> model(rows * (dim / 32) * 34);
    std::vector<float> x(max_tokens * dim), weights(rows * dim);
    for (size_t i = 0; i < x.size(); i++)
        x[i] = std::sin((float)i * 0.13f) * 0.713f;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < dim / 32; b++) {
            uint8_t *block = model.data() + (r * (dim / 32) + b) * 34;
            const __half scale = __float2half(0.0037f * (1 + (r + b) % 7));
            memcpy(block, &scale, sizeof(scale));
            for (uint32_t j = 0; j < 32; j++) {
                const int8_t q = (int8_t)((int)((r * 19 + b * 7 + j * 11) % 255) - 127);
                memcpy(block + 2 + j, &q, 1);
                weights[r * dim + b * 32 + j] = __half2float(scale) * q;
            }
        }
    }
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(x.size() * sizeof(float));
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(max_tokens * rows * sizeof(float));
    CHECK(dx && dy && ds4_gpu_tensor_write(dx, 0, x.data(), x.size() * sizeof(float)));
    float worst = 0;
    for (uint32_t tokens : {1u, 2u, 9u, 33u}) {
        CHECK(ds4_gpu_qwen4_dense_mm_tensor(dy, dx, model.data(), model.size(),
                                            0, 8, tokens, dim, rows));
        std::vector<float> actual(tokens * rows);
        CHECK(ds4_gpu_tensor_read(dy, 0, actual.data(), actual.size() * sizeof(float)));
        for (uint32_t t = 0; t < tokens; t++) {
            for (uint32_t r = 0; r < rows; r++) {
                double ref = 0;
                for (uint32_t j = 0; j < dim; j++)
                    ref += (double)x[t * dim + j] * weights[r * dim + j];
                CHECK(std::isfinite(actual[t * rows + r]));
                CHECK(std::fabs(actual[t * rows + r] - ref) < 2e-5 + 2e-6 * std::fabs(ref));
                worst = std::fmax(worst, std::fabs(actual[t * rows + r] - (float)ref));
            }
        }
    }
    ds4_gpu_tensor_free(dy); ds4_gpu_tensor_free(dx);
    printf("Qwen3.8 CUDA Q8 float input %ux%u, 1/2/9/33 tokens PASS (worst %.9g)\n", dim, rows, worst);
}

static void check_dense_multi_and_hc_rows() {
    constexpr uint32_t tokens = 3, in_dim = 32, rows0 = 7, rows1 = 5;
    std::vector<float> x((size_t)tokens * in_dim), w0((size_t)rows0 * in_dim);
    std::vector<__nv_bfloat16> w1((size_t)rows1 * in_dim);
    for (size_t i = 0; i < x.size(); i++) x[i] = ((int)(i % 19) - 9) * 0.03125f;
    for (size_t i = 0; i < w0.size(); i++) w0[i] = ((int)(i % 13) - 6) * 0.046875f;
    for (size_t i = 0; i < w1.size(); i++) w1[i] = __float2bfloat16(((int)(i % 17) - 8) * 0.0390625f);
    const uint64_t off1 = w0.size() * sizeof(float);
    std::vector<uint8_t> model(off1 + w1.size() * sizeof(__nv_bfloat16));
    memcpy(model.data(), w0.data(), off1);
    memcpy(model.data() + off1, w1.data(), w1.size() * sizeof(__nv_bfloat16));
    std::vector<float> expected0((size_t)tokens * rows0), expected1((size_t)tokens * rows1);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t r = 0; r < rows0; r++)
            for (uint32_t d = 0; d < in_dim; d++) expected0[(size_t)t * rows0 + r] += x[(size_t)t * in_dim + d] * w0[(size_t)r * in_dim + d];
        for (uint32_t r = 0; r < rows1; r++)
            for (uint32_t d = 0; d < in_dim; d++) expected1[(size_t)t * rows1 + r] += x[(size_t)t * in_dim + d] * __bfloat162float(w1[(size_t)r * in_dim + d]);
    }
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(x.size() * sizeof(float));
    ds4_gpu_tensor *do0 = ds4_gpu_tensor_alloc(expected0.size() * sizeof(float));
    ds4_gpu_tensor *do1 = ds4_gpu_tensor_alloc(expected1.size() * sizeof(float));
    CHECK(dx && do0 && do1 && ds4_gpu_tensor_write(dx, 0, x.data(), x.size() * sizeof(float)));
    ds4_gpu_tensor *outs[2] = {do0, do1};
    const uint64_t offsets[2] = {0, off1};
    const uint32_t types[2] = {0, 30}, rows[2] = {rows0, rows1};
    CHECK(ds4_gpu_qwen4_multi_gemv_tensor(dx, tokens, in_dim, 2, outs, model.data(), model.size(), offsets, types, rows));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual0(expected0.size()), actual1(expected1.size());
    CHECK(ds4_gpu_tensor_read(do0, 0, actual0.data(), actual0.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(do1, 0, actual1.data(), actual1.size() * sizeof(float)));
    float worst = 0.0f;
    for (size_t i = 0; i < actual0.size(); i++) worst = std::fmax(worst, std::fabs(actual0[i] - expected0[i]));
    for (size_t i = 0; i < actual1.size(); i++) worst = std::fmax(worst, std::fabs(actual1[i] - expected1[i]));
    CHECK(worst < 2e-5f);
    ds4_gpu_tensor_free(do1); ds4_gpu_tensor_free(do0); ds4_gpu_tensor_free(dx);

    constexpr uint32_t hc = 4, embd = 9, rank = 6;
    std::vector<float> lo((size_t)tokens * rank), u((size_t)tokens * hc * embd), xn(u.size());
    for (size_t i = 0; i < lo.size(); i++) lo[i] = ((int)(i % 9) - 4) * 0.2f;
    for (size_t i = 0; i < u.size(); i++) { u[i] = ((int)(i % 11) - 5) * 0.15f; xn[i] = ((int)(i % 7) - 3) * 0.125f; }
    std::vector<float> exp_lo(lo.size()), exp_mix((size_t)tokens * embd);
    for (size_t i = 0; i < lo.size(); i++) { const float z = lo[i] / hc; exp_lo[i] = z * sigmoid(z); }
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t d = 0; d < embd; d++) {
        for (uint32_t s = 0; s < hc; s++) { const size_t i = ((size_t)t * hc + s) * embd + d; exp_mix[(size_t)t * embd + d] += sigmoid(u[i]) * xn[i] / hc; }
    }
    auto gpu = [](const std::vector<float> &v) { ds4_gpu_tensor *p = ds4_gpu_tensor_alloc(v.size() * sizeof(float)); CHECK(p && ds4_gpu_tensor_write(p, 0, v.data(), v.size() * sizeof(float))); return p; };
    ds4_gpu_tensor *dlo = gpu(lo), *du = gpu(u), *dxn = gpu(xn);
    ds4_gpu_tensor *dlo_out = ds4_gpu_tensor_alloc(exp_lo.size() * sizeof(float));
    ds4_gpu_tensor *dmix = ds4_gpu_tensor_alloc(exp_mix.size() * sizeof(float));
    CHECK(dlo_out && dmix);
    CHECK(ds4_gpu_qwen4_hc_lo_act_tensor(dlo_out, dlo, tokens, hc, rank));
    CHECK(ds4_gpu_qwen4_hc_mix_rows_tensor(dmix, du, dxn, tokens, embd, hc));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> got_lo(exp_lo.size()), got_mix(exp_mix.size());
    CHECK(ds4_gpu_tensor_read(dlo_out, 0, got_lo.data(), got_lo.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(dmix, 0, got_mix.data(), got_mix.size() * sizeof(float)));
    for (size_t i = 0; i < got_lo.size(); i++) worst = std::fmax(worst, std::fabs(got_lo[i] - exp_lo[i]));
    for (size_t i = 0; i < got_mix.size(); i++) worst = std::fmax(worst, std::fabs(got_mix[i] - exp_mix[i]));
    CHECK(worst < 2e-5f);
    ds4_gpu_tensor_free(dmix); ds4_gpu_tensor_free(dlo_out); ds4_gpu_tensor_free(dxn); ds4_gpu_tensor_free(du); ds4_gpu_tensor_free(dlo);
    printf("Qwen3.8 CUDA dense multi-output/BF16 + HC rows PASS (worst %.9g)\n", worst);
}

static void check_mtp_stage_combine() {
    constexpr uint32_t embd = 33, hc = 4;
    constexpr float eps = 1e-6f;
    std::vector<float> e(embd), R((size_t)hc * embd), ge(embd), gh(R.size());
    for (size_t i = 0; i < e.size(); i++) { e[i] = ((int)(i % 11) - 5) * 0.13f; ge[i] = 0.8f + (i % 5) * 0.04f; }
    for (size_t i = 0; i < R.size(); i++) { R[i] = ((int)(i % 17) - 8) * 0.07f; gh[i] = 0.7f + (i % 7) * 0.03f; }
    const uint64_t gh_off = ge.size() * sizeof(float);
    std::vector<uint8_t> model(gh_off + gh.size() * sizeof(float));
    memcpy(model.data(), ge.data(), gh_off); memcpy(model.data() + gh_off, gh.data(), gh.size() * sizeof(float));
    std::vector<float> expected((size_t)(hc + 1u) * 2u * embd, 0.0f);
    float ess = 0.0f, rss = 0.0f;
    for (float v : e) ess += v * v; for (float v : R) rss += v * v;
    const float einv = 1.0f / std::sqrt(ess / embd + eps), rinv = 1.0f / std::sqrt(rss / R.size() + eps);
    for (uint32_t d = 0; d < embd; d++) expected[d] = e[d] * einv * ge[d];
    for (uint32_t s = 0; s < hc; s++) for (uint32_t d = 0; d < embd; d++)
        expected[((size_t)s + 1u) * 2u * embd + embd + d] = R[(size_t)s * embd + d] * rinv * gh[(size_t)s * embd + d];
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    auto gpu = [](const std::vector<float> &v) { ds4_gpu_tensor *p = ds4_gpu_tensor_alloc(v.size() * sizeof(float)); CHECK(p && ds4_gpu_tensor_write(p, 0, v.data(), v.size() * sizeof(float))); return p; };
    ds4_gpu_tensor *de = gpu(e), *dR = gpu(R);
    ds4_gpu_tensor *dcat = ds4_gpu_tensor_alloc(expected.size() * sizeof(float));
    CHECK(dcat && ds4_gpu_qwen4_mtp_stage_tensor(dcat, de, dR, model.data(), model.size(), 0, gh_off, embd, hc, eps));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual(expected.size());
    CHECK(ds4_gpu_tensor_read(dcat, 0, actual.data(), actual.size() * sizeof(float)));
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); i++) worst = std::fmax(worst, std::fabs(actual[i] - expected[i]));

    std::vector<float> proj((size_t)(hc + 1u) * embd), exp_out((size_t)hc * embd);
    for (size_t i = 0; i < proj.size(); i++) proj[i] = ((int)(i % 23) - 11) * 0.025f;
    for (uint32_t s = 0; s < hc; s++) for (uint32_t d = 0; d < embd; d++)
        exp_out[(size_t)s * embd + d] = proj[d] + proj[((size_t)s + 1u) * embd + d];
    ds4_gpu_tensor *dproj = gpu(proj), *dout = ds4_gpu_tensor_alloc(exp_out.size() * sizeof(float));
    CHECK(dout && ds4_gpu_qwen4_mtp_combine_tensor(dout, dproj, embd, hc));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> got(exp_out.size()); CHECK(ds4_gpu_tensor_read(dout, 0, got.data(), got.size() * sizeof(float)));
    for (size_t i = 0; i < got.size(); i++) worst = std::fmax(worst, std::fabs(got[i] - exp_out[i]));
    CHECK(worst < 2e-6f);
    ds4_gpu_tensor_free(dout); ds4_gpu_tensor_free(dproj); ds4_gpu_tensor_free(dcat); ds4_gpu_tensor_free(dR); ds4_gpu_tensor_free(de);
    printf("Qwen3.8 CUDA MTP stage/combine PASS (worst %.9g)\n", worst);
}

int main() {
    CHECK(ds4_gpu_init());
    check_hc_combine();
    check_conv_stream(false);
    check_conv_stream(true);
    check_gdn_prep();
    check_gdn_scan();
    check_gdn_out();
    check_hc_norm_gate_mix();
    check_ple_gate();
    check_ple_conv(false);
    check_ple_conv(true);
    check_router_topk();
    check_router_topk(33, 16);
    check_router_topk(256, 8);
    check_router_topk(257, 16);
    check_moe_lists_reduce();
    check_attn_prep();
    check_qsa_indexer();
    check_attention_decode();
    check_moe_q4k_mid_down();
    check_dense_q8_float_input(96, 5);
    check_dense_q8_float_input(256, 17);
    check_dense_q8_float_input(4096, 17);
    check_dense_multi_and_hc_rows();
    check_mtp_stage_combine();
    ds4_gpu_cleanup();
    return 0;
}
