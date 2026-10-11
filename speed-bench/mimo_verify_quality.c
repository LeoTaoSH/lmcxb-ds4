/* Teacher-continuation distortion probe for the pinned MOPD target.
 * Build like tests/test_deepseek41_graph: include ds4.c, omit ds4.o at link.
 * No sampling, drafter, chat template, acceptance test, or speed measurement.
 * Row i follows teacher[i] and is scored only against teacher[i + 1]. */
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "mimo_verify_quality needs finite-aware math; compile without fast/finite-only math"
#endif
#include "../ds4.c"
#include "../ds4_ssd.h"

#define VQ_NAME "mimo-verify-quality"
#define VQ_MODEL_NAME "MiMo-V2.6-Flash-MOPD-MXFP4-Q8Attn.gguf"
#define VQ_MODEL_PATH "gguf/" VQ_MODEL_NAME
#define VQ_REVISION "b12f5573350bb1704513fcf119fca48e498b0423"
#define VQ_MODEL_BYTES UINT64_C(168419206848)
#define VQ_HEADER_BYTES 96u
#define VQ_BLOCK 8u

typedef struct {
    const char *model, *prompt, *prompt_file, *tokens, *write_reference, *reference, *json;
    uint32_t limit, block, context, cache_experts;
    uint64_t cache_bytes;
    bool cold, self_test;
} vq_config;

typedef struct {
    uint32_t vocab, block, rows, prefix_count, teacher_count, context;
    uint64_t model_bytes;
} vq_reference_shape;

typedef struct {
    double nll, reference_nll, nll_delta, kl, max_drift;
    uint32_t argmax, reference_argmax;
} vq_row_metric;

typedef struct {
    uint32_t rows, evaluated_rows, agreements, prefix_count, input_teacher_count, vocab, budget;
    uint64_t candidate_checked, reference_checked, candidate_nonfinite, reference_nonfinite;
    double nll, reference_nll, nll_delta, kl, max_drift;
} vq_report;

static bool vq_fail(char *err, size_t cap, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
    return false;
}

static bool vq_host_format(void) {
    const uint32_t one = 1u;
    return sizeof(float) == 4 && sizeof(int) == 4 && FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 && *(const uint8_t *)&one == 1u;
}

/* Center both log-softmaxes before subtracting. Adding a large common offset
 * must not erase log(sum(exp(.))) from NLL or KL through cancellation. */
static bool vq_metric(const float *reference, const float *candidate, uint32_t vocab,
                       uint32_t label, vq_row_metric *out, vq_report *report) {
    if (!candidate || !out || !report || !vocab || label >= vocab) return false;
    double cmax = -DBL_MAX, rmax = -DBL_MAX;
    uint32_t ca = 0, ra = 0;
    uint64_t cbad = 0, rbad = 0;
    for (uint32_t v = 0; v < vocab; v++) {
        if (!isfinite(candidate[v])) cbad++;
        else if ((double)candidate[v] > cmax) { cmax = candidate[v]; ca = v; }
        if (reference) {
            if (!isfinite(reference[v])) rbad++;
            else if ((double)reference[v] > rmax) { rmax = reference[v]; ra = v; }
        }
    }
    report->candidate_checked += vocab;
    report->candidate_nonfinite += cbad;
    if (reference) {
        report->reference_checked += vocab;
        report->reference_nonfinite += rbad;
    }
    if (cbad || rbad) return false;
    double csum = 0.0, rsum = 0.0;
    for (uint32_t v = 0; v < vocab; v++) {
        csum += exp((double)candidate[v] - cmax);
        if (reference) rsum += exp((double)reference[v] - rmax);
    }
    const double clogsum = log(csum), rlogsum = reference ? log(rsum) : 0.0;
    *out = (vq_row_metric){.nll = (cmax - (double)candidate[label]) + clogsum,
                           .argmax = ca, .reference_argmax = ra};
    if (reference) {
        out->reference_nll = (rmax - (double)reference[label]) + rlogsum;
        out->nll_delta = (cmax - rmax) - ((double)candidate[label] - (double)reference[label]) +
                         (clogsum - rlogsum);
        for (uint32_t v = 0; v < vocab; v++) {
            const double rl = ((double)reference[v] - rmax) - rlogsum;
            const double cl = ((double)candidate[v] - cmax) - clogsum;
            out->kl += exp(rl) * (rl - cl);
            out->max_drift = fmax(out->max_drift,
                                  fabs((double)reference[v] - (double)candidate[v]));
        }
        if (out->kl < -1e-10) return false;
        if (out->kl < 0.0) out->kl = 0.0;
    }
    return isfinite(out->nll) && isfinite(out->reference_nll) && isfinite(out->nll_delta) &&
           isfinite(out->kl) && isfinite(out->max_drift);
}

static void vq_accumulate(vq_report *report, const vq_row_metric *row, bool comparison) {
    report->rows++;
    report->nll += row->nll;
    if (comparison) {
        report->reference_nll += row->reference_nll;
        report->nll_delta += row->nll_delta;
        report->kl += row->kl;
        report->agreements += row->argmax == row->reference_argmax;
        report->max_drift = fmax(report->max_drift, row->max_drift);
    }
}

static void vq_put_u32(uint8_t *dst, uint32_t n) {
    for (unsigned i = 0; i < 4; i++) dst[i] = (uint8_t)(n >> (8u * i));
}

static void vq_header(uint8_t out[VQ_HEADER_BYTES], const vq_reference_shape *shape) {
    /* Fixed header: magic[8], ten LE u32s, model bytes LE u64, revision[40].
     * Followed by prefix IDs, rows+1 teacher IDs, then row-major LE FP32 logits.
     * Exact ID arrays bind tokenization and labels; no hash collision assumption. */
    const uint32_t fields[] = {1u, 0x01020304u, 4u, shape->vocab, shape->block,
        shape->rows, shape->prefix_count, shape->teacher_count, shape->context, 0u};
    memcpy(out, "MIMOVQ1\0", 8u);
    for (unsigned i = 0; i < sizeof(fields) / sizeof(*fields); i++)
        vq_put_u32(out + 8u + 4u * i, fields[i]);
    for (unsigned i = 0; i < 8; i++) out[48u + i] = (uint8_t)(shape->model_bytes >> (8u * i));
    memcpy(out + 56u, VQ_REVISION, 40u);
}

static uint64_t vq_reference_bytes(const vq_reference_shape *shape) {
    return VQ_HEADER_BYTES + 4u * ((uint64_t)shape->prefix_count + shape->teacher_count) +
           4u * (uint64_t)shape->rows * shape->vocab;
}

static bool vq_write_header(FILE *fp, const vq_reference_shape *shape,
                             const int *prefix, const int *teacher, char *err, size_t cap) {
    uint8_t header[VQ_HEADER_BYTES];
    vq_header(header, shape);
    if (fwrite(header, 1, sizeof(header), fp) != sizeof(header) ||
        fwrite(prefix, sizeof(*prefix), shape->prefix_count, fp) != shape->prefix_count ||
        fwrite(teacher, sizeof(*teacher), shape->teacher_count, fp) != shape->teacher_count)
        return vq_fail(err, cap, "reference header/ID write failed: %s", strerror(errno));
    return true;
}

static bool vq_read_header(FILE *fp, const vq_reference_shape *shape,
                            const int *prefix, const int *teacher, char *err, size_t cap) {
    uint8_t expected[VQ_HEADER_BYTES], actual[VQ_HEADER_BYTES];
    vq_header(expected, shape);
    if (fread(actual, 1, sizeof(actual), fp) != sizeof(actual))
        return vq_fail(err, cap, "truncated/unreadable reference header");
    if (memcmp(expected, actual, sizeof(actual)))
        return vq_fail(err, cap, "incompatible reference: format/model/vocab/block/rows/context/counts differ");
    const int *ids[] = {prefix, teacher};
    const uint32_t counts[] = {shape->prefix_count, shape->teacher_count};
    for (unsigned part = 0; part < 2; part++) {
        for (uint32_t i = 0; i < counts[part]; i++) {
            int actual_id;
            if (fread(&actual_id, sizeof(actual_id), 1, fp) != 1)
                return vq_fail(err, cap, "truncated reference %s IDs", part ? "teacher" : "prefix");
            if (actual_id != ids[part][i])
                return vq_fail(err, cap, "incompatible reference %s ID at index %u", part ? "teacher" : "prefix", i);
        }
    }
    struct stat st;
    if (fstat(fileno(fp), &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uint64_t)st.st_size != vq_reference_bytes(shape))
        return vq_fail(err, cap, "reference file length differs from declared FP32 row payload");
    return true;
}

static bool vq_read_block(FILE *fp, float *rows, size_t count, char *err, size_t cap) {
    if (fread(rows, sizeof(*rows), count, fp) != count)
        return vq_fail(err, cap, "truncated/unreadable reference logit block");
    return true;
}

static bool vq_reference_end(FILE *fp, char *err, size_t cap) {
    if (fgetc(fp) != EOF || ferror(fp))
        return vq_fail(err, cap, "reference has trailing bytes or an I/O error");
    return true;
}

static bool vq_parse_uint(const char *text, uint32_t max, uint32_t *out) {
    uint32_t value = 0;
    if (!text || !*text) return false;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p < '0' || *p > '9' || value > max / 10u ||
            (value == max / 10u && (uint32_t)(*p - '0') > max % 10u)) return false;
        value = value * 10u + (uint32_t)(*p - '0');
    }
    *out = value;
    return true;
}

static bool vq_parse_teacher(const char *text, ds4_tokens *out, char *err, size_t cap) {
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        while (isspace(*p)) p++;
        if (!*p) break;
        uint32_t value = 0;
        const unsigned char *start = p;
        while (*p >= '0' && *p <= '9') {
            const uint32_t digit = (uint32_t)(*p - '0');
            if (value > (uint32_t)INT_MAX / 10u ||
                (value == (uint32_t)INT_MAX / 10u && digit > (uint32_t)INT_MAX % 10u))
                return vq_fail(err, cap, "teacher ID %d exceeds INT_MAX", out->len);
            value = value * 10u + digit;
            p++;
        }
        if (p == start || (*p && !isspace(*p)))
            return vq_fail(err, cap, "teacher IDs must be whitespace-separated nonnegative decimal integers (index %d)", out->len);
        if (out->len >= INT_MAX / 2)
            return vq_fail(err, cap, "too many teacher IDs");
        token_vec_push(out, (int)value);
    }
    return true;
}

static bool vq_read_text(const char *path, char **out, char *err, size_t cap) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return vq_fail(err, cap, "cannot open %s: %s", path, strerror(errno));
    struct stat st;
    bool ok = !fstat(fileno(fp), &st) && S_ISREG(st.st_mode) && st.st_size >= 0 &&
              (uint64_t)st.st_size < SIZE_MAX;
    char *text = NULL;
    if (ok) {
        const size_t size = (size_t)st.st_size;
        text = xmalloc(size + 1u);
        ok = fread(text, 1, size, fp) == size && !memchr(text, '\0', size);
        text[size] = '\0';
    }
    if (fclose(fp)) ok = false;
    if (!ok) {
        free(text);
        return vq_fail(err, cap, "cannot read NUL-free regular text file %s", path);
    }
    *out = text;
    return true;
}

static void vq_json_string(FILE *fp, const char *text) {
    if (!text) { fputs("null", fp); return; }
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', fp); fputc(*p, fp); }
        else if (*p < 0x20u) fprintf(fp, "\\u%04x", (unsigned)*p);
        else fputc(*p, fp);
    }
    fputc('"', fp);
}

static void vq_json_number(FILE *fp, double number, bool available) {
    if (available) fprintf(fp, "%.17g", number);
    else fputs("null", fp);
}

static bool vq_write_json(FILE *fp, const vq_config *cfg, const vq_report *r,
                           const vq_row_metric *rows, const ds4_tokens *teacher,
                           bool success, const char *err) {
    const bool comparison = cfg->reference != NULL, measured = r->rows != 0;
    const double divisor = measured ? r->rows : 1u;
    fprintf(fp, "{\n  \"probe\": \"teacher-continuation-distortion\",\n  \"success\": %s,\n  \"error\": ", success ? "true" : "false");
    vq_json_string(fp, success ? NULL : err);
    fputs(",\n  \"mode\": ", fp); vq_json_string(fp, comparison ? "comparison" : "write-reference");
    fputs(",\n  \"model\": ", fp); vq_json_string(fp, cfg->model);
    fputs(",\n  \"pinned_source_revision\": \"" VQ_REVISION "\",\n  \"model_file_bytes\": ", fp);
    fprintf(fp, "%" PRIu64, VQ_MODEL_BYTES);
    fputs(",\n  \"prompt_file\": ", fp); vq_json_string(fp, cfg->prompt_file);
    fputs(",\n  \"token_file\": ", fp); vq_json_string(fp, cfg->tokens);
    fputs(",\n  \"reference_file\": ", fp); vq_json_string(fp, comparison ? cfg->reference : cfg->write_reference);
    fprintf(fp, ",\n  \"vocab\": %u,\n  \"block\": %u,\n  \"requested_rows\": %u,\n  \"evaluated_rows\": %u,\n  \"scored_rows\": %u,\n  \"prefix_tokens\": %u,\n  \"teacher_ids_in_file\": %u,\n  \"teacher_ids_used\": %u,\n  \"label_offset\": 1,\n  \"context\": %u,\n  \"expert_budget\": %u,\n  \"cache_experts_requested\": %u,\n  \"cache_bytes_requested\": %" PRIu64 ",\n  \"ssd_streaming_cold\": %s,\n  \"expert_rows_env\": ",
            r->vocab, cfg->block, cfg->limit, r->evaluated_rows, r->rows, r->prefix_count,
            r->input_teacher_count, cfg->limit + 1u, cfg->context, r->budget,
            cfg->cache_experts, cfg->cache_bytes, cfg->cold ? "true" : "false");
    vq_json_string(fp, getenv("DS4_MIMO2_EXPERT_ROWS"));
    fprintf(fp, ",\n  \"finite_checks\": {\"candidate_values\": %" PRIu64 ", \"reference_values\": %" PRIu64 ", \"candidate_nonfinite\": %" PRIu64 ", \"reference_nonfinite\": %" PRIu64 "},\n  \"metric_units\": \"nats\",\n  \"teacher_mean_nll\": ",
            r->candidate_checked, r->reference_checked, r->candidate_nonfinite, r->reference_nonfinite);
    vq_json_number(fp, r->nll / divisor, measured);
    fputs(",\n  \"reference_teacher_mean_nll\": ", fp);
    vq_json_number(fp, r->reference_nll / divisor, comparison && measured);
    fputs(",\n  \"teacher_mean_nll_delta\": ", fp);
    vq_json_number(fp, r->nll_delta / divisor, comparison && measured);
    fputs(",\n  \"mean_kl_reference_to_candidate\": ", fp);
    vq_json_number(fp, r->kl / divisor, comparison && measured);
    fprintf(fp, ",\n  \"argmax_agreements\": %u,\n  \"argmax_agreement\": ", r->agreements);
    vq_json_number(fp, r->agreements / divisor, comparison && measured);
    fputs(",\n  \"max_logit_drift\": ", fp);
    vq_json_number(fp, r->max_drift, comparison && measured);
    fputs(",\n  \"rows\": [", fp);
    for (uint32_t i = 0; rows && i < r->rows; i++) {
        const vq_row_metric *m = &rows[i];
        fprintf(fp, "%s\n    {\"post_teacher_index\": %u, \"input_id\": %d, \"label_index\": %u, \"label_id\": %d, \"nll\": %.17g, \"argmax\": %u",
                i ? "," : "", i, teacher->v[i], i + 1u, teacher->v[i + 1u], m->nll, m->argmax);
        if (comparison)
            fprintf(fp, ", \"reference_nll\": %.17g, \"nll_delta\": %.17g, \"kl_reference_to_candidate\": %.17g, \"reference_argmax\": %u, \"max_logit_drift\": %.17g",
                    m->reference_nll, m->nll_delta, m->kl, m->reference_argmax, m->max_drift);
        fputc('}', fp);
    }
    fputs("\n  ]\n}\n", fp);
    return !ferror(fp);
}

static void vq_usage(FILE *fp, const char *program) {
    fprintf(fp, "Usage: %s --prompt-file FILE --tokens FILE [options]\n"
        "  --model FILE          pinned MOPD MXFP4-Q8Attn GGUF (default " VQ_MODEL_PATH ")\n"
        "  --prompt TEXT         raw prefix, instead of --prompt-file\n"
        "  --prompt-file FILE    raw prefix; recognizes rendered special tokens, adds no template/BOS\n"
        "  --tokens FILE         whitespace-separated teacher IDs; at least limit+1 IDs\n"
        "  --limit N             scored post-token rows (default 128; positive multiple of 8)\n"
        "  --block N             fixed verifier block (only 8; default 8)\n"
        "  --ctx N               context capacity (default 4096)\n"
        "  --ssd-streaming       enabled always for this MOPD probe\n"
        "  --ssd-streaming-cold   skip expert preloading\n"
        "  --ssd-streaming-cache-experts N|NGB  cache budget (default automatic)\n"
        "  --write-reference FILE  stream full-vocab FP32 rows; budget env must be unset\n"
        "  --reference FILE      compare one block at a time against an exact reference\n"
        "  --json FILE           write metrics JSON (output files must not already exist)\n"
        "  --self-test           CPU-only metric and reference-format tests; opens no model\n"
        "Set DS4_MIMO2_VERIFY_EXPERT_BUDGET=1..256 only for a comparison pilot.\n"
        "Default exact path: DS4_MIMO2_EXPERT_ROWS=1 when unset.\n"
        "This measures teacher-continuation distortion, not task accuracy or speed.\n", program);
}

static bool vq_parse_options(int argc, char **argv, vq_config *cfg, char *err, size_t cap) {
    *cfg = (vq_config){.model = VQ_MODEL_PATH, .limit = 128u, .block = VQ_BLOCK, .context = 4096u};
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--self-test")) { cfg->self_test = true; continue; }
        if (!strcmp(arg, "--ssd-streaming")) continue;
        if (!strcmp(arg, "--ssd-streaming-cold")) { cfg->cold = true; continue; }
        if (i + 1 >= argc) return vq_fail(err, cap, "missing value for %s", arg);
        const char *value = argv[++i];
        if (!strcmp(arg, "--model") || !strcmp(arg, "-m")) cfg->model = value;
        else if (!strcmp(arg, "--prompt")) cfg->prompt = value;
        else if (!strcmp(arg, "--prompt-file")) cfg->prompt_file = value;
        else if (!strcmp(arg, "--tokens") || !strcmp(arg, "--token-file")) cfg->tokens = value;
        else if (!strcmp(arg, "--write-reference")) cfg->write_reference = value;
        else if (!strcmp(arg, "--reference")) cfg->reference = value;
        else if (!strcmp(arg, "--json")) cfg->json = value;
        else if (!strcmp(arg, "--limit") || !strcmp(arg, "--block") ||
                 !strcmp(arg, "--ctx") || !strcmp(arg, "--context")) {
            uint32_t parsed;
            if (!vq_parse_uint(value, INT_MAX - 1u, &parsed) || !parsed)
                return vq_fail(err, cap, "invalid positive integer for %s: %s", arg, value);
            if (!strcmp(arg, "--limit")) cfg->limit = parsed;
            else if (!strcmp(arg, "--block")) cfg->block = parsed;
            else cfg->context = parsed;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            if (!ds4_parse_streaming_cache_experts_arg(value, &cfg->cache_experts, &cfg->cache_bytes))
                return vq_fail(err, cap, "invalid expert-cache budget: %s", value);
        } else return vq_fail(err, cap, "unknown option: %s", arg);
    }
    if (cfg->block != VQ_BLOCK || cfg->limit % VQ_BLOCK)
        return vq_fail(err, cap, "--block must be 8 and --limit a positive multiple of 8");
    if (cfg->self_test) return true;
    if ((!cfg->prompt == !cfg->prompt_file) || !cfg->tokens ||
        (!cfg->write_reference == !cfg->reference))
        return vq_fail(err, cap, "require exactly one raw prompt source, --tokens, and exactly one reference mode");
    return true;
}

static FILE *vq_test_reference(const vq_reference_shape *shape, const int *prefix,
                                const int *teacher, const float *rows) {
    char err[256];
    FILE *fp = tmpfile();
    const size_t count = (size_t)shape->rows * shape->vocab;
    if (!fp) return NULL;
    if (setvbuf(fp, NULL, _IONBF, 0)) { fclose(fp); return NULL; }
    if (!vq_write_header(fp, shape, prefix, teacher, err, sizeof(err)) ||
        fwrite(rows, sizeof(*rows), count, fp) != count || fflush(fp) || fseek(fp, 0, SEEK_SET)) {
        fclose(fp); return NULL;
    }
    return fp;
}

static int vq_self_test(void) {
    int rc = 1;
    FILE *fp = NULL;
    ds4_tokens parsed = {0};
    char err[256];
    vq_report report = {0};
    vq_row_metric metric;
#define VQ_REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, VQ_NAME ": self-test failed at line %d: %s\n", __LINE__, #condition); goto done; \
} } while (0)
    const float uniform[] = {0, 0}, peaked[] = {0, (float)log(3.0)};
    const float shifted[] = {1000000, 1000000}, extreme[] = {0, -10000};
    VQ_REQUIRE(vq_host_format());
    VQ_REQUIRE(vq_metric(uniform, uniform, 2, 1, &metric, &report));
    VQ_REQUIRE(fabs(metric.nll - log(2.0)) < 1e-12 && metric.kl == 0 && metric.max_drift == 0);
    VQ_REQUIRE(vq_metric(uniform, shifted, 2, 1, &metric, &report));
    VQ_REQUIRE(fabs(metric.nll - log(2.0)) < 1e-12 && metric.kl == 0 && metric.max_drift == 1000000);
    VQ_REQUIRE(vq_metric(peaked, uniform, 2, 1, &metric, &report));
    VQ_REQUIRE(fabs(metric.kl - (0.25 * log(0.5) + 0.75 * log(1.5))) < 1e-7);
    VQ_REQUIRE(fabs(metric.nll_delta - log(1.5)) < 1e-7);
    VQ_REQUIRE(metric.argmax == 0 && metric.reference_argmax == 1);
    VQ_REQUIRE(vq_metric(uniform, extreme, 2, 1, &metric, &report));
    VQ_REQUIRE(fabs(metric.nll - 10000) < 1e-12 && fabs(metric.kl - (5000 - log(2.0))) < 1e-9);
    const float huge_reference[] = {0, 0, -FLT_MAX}, huge_candidate[] = {0, -10000, -FLT_MAX};
    VQ_REQUIRE(vq_metric(huge_reference, huge_candidate, 3, 2, &metric, &report));
    VQ_REQUIRE(fabs(metric.nll_delta + log(2.0)) < 1e-12);
    vq_accumulate(&report, &metric, true);
    VQ_REQUIRE(fabs(report.nll_delta + log(2.0)) < 1e-12);
    VQ_REQUIRE(!vq_metric(uniform, extreme, 2, 2, &metric, &report));
    const float nan_row[] = {NAN, 0}, inf_row[] = {0, INFINITY};
    VQ_REQUIRE(!vq_metric(uniform, nan_row, 2, 1, &metric, &report));
    VQ_REQUIRE(!vq_metric(inf_row, uniform, 2, 1, &metric, &report));
    VQ_REQUIRE(report.candidate_nonfinite == 1 && report.reference_nonfinite == 1);
    VQ_REQUIRE(vq_parse_teacher("0\n1 0\t1", &parsed, err, sizeof(err)) && parsed.len == 4);
    VQ_REQUIRE(vq_metric(NULL, peaked, 2, (uint32_t)parsed.v[1], &metric, &report));
    VQ_REQUIRE(fabs(metric.nll - log(4.0 / 3.0)) < 1e-7); /* label i+1, not input i */
    VQ_REQUIRE(!vq_parse_teacher("-1", &parsed, err, sizeof(err)));
    VQ_REQUIRE(!vq_parse_teacher("2147483648", &parsed, err, sizeof(err)));
    VQ_REQUIRE(!vq_parse_teacher("1,2", &parsed, err, sizeof(err)));
    const int prefix[] = {1, 0}, teacher[] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
    const float rows[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    float loaded[16];
    const vq_reference_shape shape = {2, 8, 8, 2, 9, 4096, VQ_MODEL_BYTES};
    fp = vq_test_reference(&shape, prefix, teacher, rows);
    VQ_REQUIRE(fp && vq_read_header(fp, &shape, prefix, teacher, err, sizeof(err)));
    VQ_REQUIRE(vq_read_block(fp, loaded, 16, err, sizeof(err)) && !memcmp(loaded, rows, sizeof(rows)));
    VQ_REQUIRE(vq_reference_end(fp, err, sizeof(err)));
    fclose(fp); fp = NULL;
    const long mismatch_offsets[] = {0, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56,
                                     VQ_HEADER_BYTES, VQ_HEADER_BYTES + 8};
    for (unsigned i = 0; i < sizeof(mismatch_offsets) / sizeof(*mismatch_offsets); i++) {
        fp = vq_test_reference(&shape, prefix, teacher, rows);
        VQ_REQUIRE(fp && !fseek(fp, mismatch_offsets[i], SEEK_SET));
        const int old = fgetc(fp);
        VQ_REQUIRE(old != EOF && !fseek(fp, mismatch_offsets[i], SEEK_SET));
        VQ_REQUIRE(fputc(old ^ 0x01, fp) != EOF && !fflush(fp) && !fseek(fp, 0, SEEK_SET));
        VQ_REQUIRE(!vq_read_header(fp, &shape, prefix, teacher, err, sizeof(err)));
        fclose(fp); fp = NULL;
    }
    const off_t short_sizes[] = {7, VQ_HEADER_BYTES - 1, VQ_HEADER_BYTES + 3,
                                 (off_t)(VQ_HEADER_BYTES + 44 + sizeof(rows) - 1)};
    for (unsigned i = 0; i < sizeof(short_sizes) / sizeof(*short_sizes); i++) {
        fp = vq_test_reference(&shape, prefix, teacher, rows);
        VQ_REQUIRE(fp && !ftruncate(fileno(fp), short_sizes[i]));
        VQ_REQUIRE(!vq_read_header(fp, &shape, prefix, teacher, err, sizeof(err)));
        fclose(fp); fp = NULL;
    }
    fp = vq_test_reference(&shape, prefix, teacher, rows);
    VQ_REQUIRE(fp && !fseek(fp, 0, SEEK_END) && fputc(0, fp) != EOF && !fflush(fp) && !fseek(fp, 0, SEEK_SET));
    VQ_REQUIRE(!vq_read_header(fp, &shape, prefix, teacher, err, sizeof(err)));
    fclose(fp); fp = vq_test_reference(&shape, prefix, teacher, rows);
    VQ_REQUIRE(fp && vq_read_header(fp, &shape, prefix, teacher, err, sizeof(err)));
    VQ_REQUIRE(!ftruncate(fileno(fp), (off_t)(vq_reference_bytes(&shape) - 1u)));
    /* Test a truncation after header validation with unbuffered I/O. */
    VQ_REQUIRE(!vq_read_block(fp, loaded, 16, err, sizeof(err)));
    rc = 0;
    puts(VQ_NAME ": self-test PASS (metrics, next-token labels, finite rejection, header/ID binding, truncation/trailing bytes)");
done:
    if (fp) fclose(fp);
    ds4_tokens_free(&parsed);
#undef VQ_REQUIRE
    return rc;
}

static int vq_run(const vq_config *cfg) {
    int rc = 1;
    char err[512] = "";
    char *prefix_text = NULL, *teacher_text = NULL;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prefix = {0}, teacher = {0};
    vq_report report = {0};
    vq_row_metric *metrics = NULL;
    float *candidate = NULL, *reference = NULL;
    FILE *reference_fp = NULL, *json_fp = NULL;
    bool wrote_reference = false;
    const bool comparison = cfg->reference != NULL;
#define VQ_TRY(condition) do { if (!(condition)) goto done; } while (0)
    if (cfg->json) {
        json_fp = fopen(cfg->json, "wx");
        if (!json_fp) {
            fprintf(stderr, VQ_NAME ": cannot create JSON %s: %s\n", cfg->json, strerror(errno));
            return 1;
        }
    }
    VQ_TRY(vq_host_format() || vq_fail(err, sizeof(err), "requires little-endian IEEE FP32 and 32-bit int"));
    const char *budget_env = getenv("DS4_MIMO2_VERIFY_EXPERT_BUDGET");
    if (budget_env) {
        VQ_TRY(comparison || vq_fail(err, sizeof(err), "unset DS4_MIMO2_VERIFY_EXPERT_BUDGET when writing an exact reference"));
        VQ_TRY(vq_parse_uint(budget_env, 256u, &report.budget) && report.budget > 0u ||
               vq_fail(err, sizeof(err), "DS4_MIMO2_VERIFY_EXPERT_BUDGET must be in 1..256"));
    }
    if (!getenv("DS4_MIMO2_EXPERT_ROWS"))
        VQ_TRY(!setenv("DS4_MIMO2_EXPERT_ROWS", "1", 0) || vq_fail(err, sizeof(err), "cannot select exact shared-row path"));
    struct stat model_stat;
    const char *basename = strrchr(cfg->model, '/');
    basename = basename ? basename + 1 : cfg->model;
    VQ_TRY(!strcmp(basename, VQ_MODEL_NAME) && !stat(cfg->model, &model_stat) &&
           S_ISREG(model_stat.st_mode) && model_stat.st_size >= 0 &&
           (uint64_t)model_stat.st_size == VQ_MODEL_BYTES ||
           vq_fail(err, sizeof(err), "--model must name the pinned %s (%" PRIu64 " bytes)", VQ_MODEL_NAME, VQ_MODEL_BYTES));
    VQ_TRY(vq_read_text(cfg->tokens, &teacher_text, err, sizeof(err)));
    VQ_TRY(vq_parse_teacher(teacher_text, &teacher, err, sizeof(err)));
    report.input_teacher_count = (uint32_t)teacher.len;
    VQ_TRY((uint32_t)teacher.len >= cfg->limit + 1u ||
           vq_fail(err, sizeof(err), "need %u teacher IDs for %u post-token rows; got %d", cfg->limit + 1u, cfg->limit, teacher.len));
    if (cfg->prompt_file) VQ_TRY(vq_read_text(cfg->prompt_file, &prefix_text, err, sizeof(err)));
    VQ_TRY(cfg->limit < cfg->context || vq_fail(err, sizeof(err), "context must fit nonempty prefix plus limit rows"));
#ifdef DS4_HAS_MIMO2_METAL
    const ds4_engine_options options = {.model_path = cfg->model, .backend = DS4_BACKEND_METAL,
        .context_size = (int)cfg->context, .ssd_streaming = true, .ssd_streaming_cold = cfg->cold,
        .ssd_streaming_cache_experts = cfg->cache_experts, .ssd_streaming_cache_bytes = cfg->cache_bytes};
    VQ_TRY(!ds4_engine_open(&engine, &options) || vq_fail(err, sizeof(err), "model open failed"));
    VQ_TRY(ds4_engine_is_mimo2(engine) && engine->model.size == VQ_MODEL_BYTES ||
           vq_fail(err, sizeof(err), "model is not the pinned MiMo target"));
    report.vocab = DS4_N_VOCAB;
    ds4_tokenize_text(engine, cfg->prompt ? cfg->prompt : prefix_text, &prefix);
    report.prefix_count = (uint32_t)prefix.len;
    VQ_TRY(prefix.len > 0 && (uint32_t)prefix.len <= cfg->context - cfg->limit ||
           vq_fail(err, sizeof(err), "prefix (%d) plus %u teacher rows does not fit context %u", prefix.len, cfg->limit, cfg->context));
    for (int i = 0; i < teacher.len; i++)
        VQ_TRY((uint32_t)teacher.v[i] < report.vocab ||
               vq_fail(err, sizeof(err), "teacher ID %d at index %d is outside vocabulary %u", teacher.v[i], i, report.vocab));
    const vq_reference_shape shape = {report.vocab, cfg->block, cfg->limit, report.prefix_count,
                                      cfg->limit + 1u, cfg->context, engine->model.size};
    reference_fp = fopen(comparison ? cfg->reference : cfg->write_reference, comparison ? "rb" : "wx");
    VQ_TRY(reference_fp || vq_fail(err, sizeof(err), "cannot %s reference: %s", comparison ? "open" : "create", strerror(errno)));
    wrote_reference = !comparison;
    if (comparison) VQ_TRY(vq_read_header(reference_fp, &shape, prefix.v, teacher.v, err, sizeof(err)));
    else VQ_TRY(vq_write_header(reference_fp, &shape, prefix.v, teacher.v, err, sizeof(err)));
    const size_t block_values = (size_t)cfg->block * report.vocab;
    candidate = xmalloc(block_values * sizeof(*candidate));
    if (comparison) reference = xmalloc(block_values * sizeof(*reference));
    if (json_fp) metrics = xcalloc(cfg->limit, sizeof(*metrics));
    VQ_TRY(!ds4_session_create(&session, engine, (int)cfg->context) || vq_fail(err, sizeof(err), "session create failed"));
    VQ_TRY(!ds4_session_sync(session, &prefix, err, sizeof(err)));
    VQ_TRY(session->mimo2_graph_ready && session->mimo2_graph.prefill_cap >= cfg->block &&
           !session->mimo2_graph.dflash || vq_fail(err, sizeof(err), "native verifier graph unavailable"));
    const int total_tokens = prefix.len + (int)cfg->limit;
    if (session->checkpoint.cap < total_tokens) {
        session->checkpoint.v = xrealloc(session->checkpoint.v, (size_t)total_tokens * sizeof(*session->checkpoint.v));
        session->checkpoint.cap = total_tokens;
    }
    for (uint32_t start = 0; start < cfg->limit; start += cfg->block) {
        ds4_mimo2_gpu_graph *graph = &session->mimo2_graph;
        VQ_TRY(session->checkpoint_valid && graph->pos == (uint32_t)session->checkpoint.len ||
               vq_fail(err, sizeof(err), "checkpoint/KV frontier differs before row %u", start));
        if (comparison) VQ_TRY(vq_read_block(reference_fp, reference, block_values, err, sizeof(err)));
        session->checkpoint_valid = false;
        VQ_TRY(mimo2_graph_verify_rows(graph, &engine->model, &engine->weights,
                                      teacher.v + start, cfg->block, NULL, candidate) ||
               vq_fail(err, sizeof(err), "native verifier failed at teacher row %u", start));
        /* Every supplied input token commits, irrespective of its probability.
         * No rejection/rewind occurs, so no speculative ring backup is needed. */
        memcpy(session->checkpoint.v + session->checkpoint.len, teacher.v + start,
               cfg->block * sizeof(*teacher.v));
        session->checkpoint.len += (int)cfg->block;
        memcpy(session->logits, candidate + (size_t)(cfg->block - 1u) * report.vocab,
               (size_t)report.vocab * sizeof(*candidate));
        mimo2_session_capture_hidden(session, cfg->block);
        session->checkpoint_valid = true;
        session->mtp_draft_valid = false;
        report.evaluated_rows += cfg->block;
        VQ_TRY(graph->pos == (uint32_t)session->checkpoint.len ||
               vq_fail(err, sizeof(err), "checkpoint/KV frontier differs after row %u", start));
        for (uint32_t i = 0; i < cfg->block; i++) {
            vq_row_metric row;
            VQ_TRY(vq_metric(comparison ? reference + (size_t)i * report.vocab : NULL,
                              candidate + (size_t)i * report.vocab, report.vocab,
                              (uint32_t)teacher.v[start + i + 1u], &row, &report) ||
                   vq_fail(err, sizeof(err), "nonfinite logits/metric at teacher row %u", start + i));
            if (metrics) metrics[start + i] = row;
            vq_accumulate(&report, &row, comparison);
        }
        if (!comparison)
            VQ_TRY(fwrite(candidate, sizeof(*candidate), block_values, reference_fp) == block_values ||
                   vq_fail(err, sizeof(err), "reference row write failed at row %u", start));
    }
    if (comparison) VQ_TRY(vq_reference_end(reference_fp, err, sizeof(err)));
    if (fclose(reference_fp)) {
        reference_fp = NULL;
        vq_fail(err, sizeof(err), "reference close failed: %s", strerror(errno));
        goto done;
    }
    reference_fp = NULL;
    rc = 0;
#else
    vq_fail(err, sizeof(err), "model-backed probe requires the MiMo Metal build; --self-test is model-free");
#endif
done:
    if (reference_fp && fclose(reference_fp)) rc = 1;
    if (rc && wrote_reference) unlink(cfg->write_reference);
    if (json_fp) {
        const bool json_ok = vq_write_json(json_fp, cfg, &report, metrics, &teacher, rc == 0, err);
        const int close_result = fclose(json_fp);
        if (!json_ok || close_result) {
            fprintf(stderr, VQ_NAME ": JSON write/close failed\n");
            rc = 1;
        }
    }
    if (rc) fprintf(stderr, VQ_NAME ": %s\n", err[0] ? err : "I/O failure");
    else {
        printf(VQ_NAME ": teacher-continuation distortion, rows=%u block=%u budget=%u finite=%" PRIu64 "/%" PRIu64 "\n",
               report.rows, cfg->block, report.budget, report.candidate_checked, report.reference_checked);
        printf("teacher NLL %.9g nats", report.nll / report.rows);
        if (comparison)
            printf("; reference %.9g; delta %+.9g; KL(ref||candidate) %.9g; argmax %u/%u (%.6f); max logit drift %.9g",
                   report.reference_nll / report.rows, report.nll_delta / report.rows,
                   report.kl / report.rows, report.agreements, report.rows,
                   (double)report.agreements / report.rows, report.max_drift);
        putchar('\n');
    }
    if (session) ds4_session_free(session);
    if (engine) ds4_engine_close(engine);
    ds4_tokens_free(&prefix);
    ds4_tokens_free(&teacher);
    free(prefix_text); free(teacher_text); free(metrics); free(candidate); free(reference);
#undef VQ_TRY
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        vq_usage(stdout, argv[0]); return 0;
    }
    vq_config cfg;
    char err[512];
    if (!vq_parse_options(argc, argv, &cfg, err, sizeof(err))) {
        fprintf(stderr, VQ_NAME ": %s\n", err);
        vq_usage(stderr, argv[0]);
        return 2;
    }
    return cfg.self_test ? vq_self_test() : vq_run(&cfg);
}
