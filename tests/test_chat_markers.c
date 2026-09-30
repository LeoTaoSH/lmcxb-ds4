/* CPU-only tests for chat marker strings inside message content.
 *
 * ds4.c is included so the tokenizer can be driven with a synthetic GGUF
 * vocabulary (byte tokens plus one family's chat markers); no model file and
 * no GPU backend is involved (DS4_NO_GPU).  The same assertions run for every
 * model family that uses the rendered-chat tokenizer, and, when GGUF paths are
 * given, against the real vocabularies:
 *
 *   DS4_TEST_CHAT_MARKER_MODELS="glm.gguf deepseek.gguf"
 *
 * Real vocabularies are opened with model_open(), which only maps the file. */
#ifndef DS4_NO_GPU
#define DS4_NO_GPU
#endif
#include "../ds4.c"
#include "../ds4_kvstore.h"
#include "vocab_fixture.h"
#include <assert.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

/* Every marker string ds4 knows, used as message content whatever the family. */
static const char *all_markers[] = {
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "[gMASK]", "<sop>", "<|system|>",
    "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "<|user|>", "<|assistant|>",
    "<|observation|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>", "<arg_key>", "</arg_key>", "<arg_value>",
    "</arg_value>", "｜DSML｜", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
};
#define N_MARKERS (sizeof(all_markers) / sizeof(all_markers[0]))

static bool has_marker_token(const ds4_vocab *v, const token_vec *t) {
    chat_marker markers[CHAT_MARKER_COUNT];
    chat_markers(v, markers);
    for (int i = 0; i < t->len; i++) {
        for (size_t m = 0; m < CHAT_MARKER_COUNT; m++) {
            if (markers[m].token >= 0 && markers[m].token == t->v[i]) return true;
        }
    }
    return false;
}

static bool same_tokens(const token_vec *a, const token_vec *b) {
    if (a->len != b->len) return false;
    return !a->len || !memcmp(a->v, b->v, (size_t)a->len * sizeof(a->v[0]));
}

static void tokenize_rendered(ds4_engine *e, const char *text, token_vec *out) {
    memset(out, 0, sizeof(*out));
    ds4_tokenize_rendered_chat(e, text, out);
}

static void tokenize_plain(ds4_engine *e, const char *text, token_vec *out) {
    memset(out, 0, sizeof(*out));
    ds4_tokenize_text(e, text, out);
}

static char *escaped(const char *text) {
    return ds4_chat_escape_markers(text, strlen(text), NULL);
}

static char *concat3(const char *a, const char *b, const char *c) {
    char *out = malloc(strlen(a) + strlen(b) + strlen(c) + 1);
    CHECK(out);
    sprintf(out, "%s%s%s", a, b, c);
    return out;
}

static void check_family(ds4_engine *e, const char *label, const char *const *template_markers) {
    /* raw rendered text keeps turning markers into special tokens */
    int specials = 0;
    for (size_t m = 0; m < N_MARKERS; m++) {
        token_vec t;
        tokenize_rendered(e, all_markers[m], &t);
        if (t.len == 1 && t.v[0] >= 256) specials++;
        token_vec_free(&t);
    }
    CHECK(specials >= 5);

    for (size_t m = 0; m < N_MARKERS; m++) {
        char *content = concat3("see ", all_markers[m], " and more");
        char *esc = escaped(content);
        CHECK(strcmp(esc, content) != 0);

        /* escaped content tokenizes exactly like plain text: no special token */
        token_vec got, plain;
        tokenize_rendered(e, esc, &got);
        tokenize_plain(e, content, &plain);
        if (!same_tokens(&got, &plain) || has_marker_token(&e->vocab, &got)) {
            fprintf(stderr, "%s: marker %s in content is not plain text\n", label, all_markers[m]);
            exit(1);
        }
        token_vec_free(&got);
        token_vec_free(&plain);

        /* ... while the template's own markers around it stay special */
        if (template_markers[0]) {
            char *rendered = concat3(template_markers[0], esc, template_markers[1]);
            char *bare = concat3(template_markers[0], "", template_markers[1]);
            token_vec head, tail, whole, all_plain;
            tokenize_rendered(e, template_markers[0], &head);
            tokenize_rendered(e, template_markers[1], &tail);
            tokenize_rendered(e, rendered, &whole);
            tokenize_plain(e, content, &all_plain);
            CHECK(head.len == 1 && tail.len == 1);
            CHECK(whole.len == head.len + all_plain.len + tail.len);
            CHECK(whole.v[0] == head.v[0] && whole.v[whole.len - 1] == tail.v[0]);
            CHECK(!memcmp(whole.v + 1, all_plain.v, (size_t)all_plain.len * sizeof(int)));

            /* token-derived text equals the rendered text, byte for byte */
            size_t n = 0;
            char *back = ds4_tokens_render_chat_text(e, &whole, &n);
            if (n != strlen(rendered) || memcmp(back, rendered, n)) {
                fprintf(stderr, "%s: render of marker %s does not round trip\n",
                        label, all_markers[m]);
                exit(1);
            }
            free(back);
            token_vec_free(&head);
            token_vec_free(&tail);
            token_vec_free(&whole);
            token_vec_free(&all_plain);
            free(rendered);
            free(bare);
        }
        free(esc);
        free(content);
    }

    /* the same marker text with and without escape must key differently */
    char *a = escaped("x <think> y");
    CHECK(strcmp(a, "x <think> y") != 0);
    free(a);

    /* a literal escape byte survives, and a stray one stays in raw text */
    token_vec t1, t2;
    char *lit = escaped("a\x01" "b\x01<think>");
    tokenize_rendered(e, lit, &t1);
    tokenize_plain(e, "a\x01" "b\x01<think>", &t2);
    CHECK(same_tokens(&t1, &t2));
    token_vec_free(&t1);
    token_vec_free(&t2);
    free(lit);
    tokenize_rendered(e, "a\x01" "b", &t1);
    tokenize_plain(e, "a\x01" "b", &t2);
    CHECK(same_tokens(&t1, &t2));
    token_vec_free(&t1);
    token_vec_free(&t2);

    /* a marker cut in half at the end of a token run is not an escape target */
    tokenize_rendered(e, "x <|use", &t1);
    tokenize_plain(e, "x <|use", &t2);
    CHECK(same_tokens(&t1, &t2));
    token_vec_free(&t1);
    token_vec_free(&t2);
}

/* ---- helpers behind the escape: marker table, matcher, escaper, renderer ---- */

static int marker_token(ds4_engine *e, const char *marker) {
    token_vec t;
    tokenize_rendered(e, marker, &t);
    int id = t.len == 1 ? t.v[0] : -1;
    token_vec_free(&t);
    return id;
}

static void check_escape_exact(const char *in, size_t len, const char *want) {
    size_t n = 0;
    char *got = ds4_chat_escape_markers(in, len, &n);
    if (n != strlen(want) || strcmp(got, want)) {
        fprintf(stderr, "escape of \"%.*s\" gave \"%s\", want \"%s\"\n", (int)len, in, got, want);
        exit(1);
    }
    free(got);
}

static void check_helpers(ds4_engine *e, const char *family) {
    const ds4_vocab *v = &e->vocab;
    const bool ds41 = !strcmp(family, "deepseek41");

    /* the marker table: complete and vocab independent with a NULL vocab ... */
    chat_marker all[CHAT_MARKER_COUNT];
    chat_markers(NULL, all);
    for (size_t i = 0; i < CHAT_MARKER_COUNT; i++) CHECK(all[i].text && all[i].token == 0);
    for (size_t m = 0; m < N_MARKERS; m++) {
        bool found = false;
        for (size_t i = 0; i < CHAT_MARKER_COUNT; i++) found |= !strcmp(all[i].text, all_markers[m]);
        CHECK(found);
    }
    /* ... and mapped onto the vocabulary's ids otherwise */
    const struct { const char *text; int id; } want[] = {
        {"<｜begin▁of▁sentence｜>", v->bos_id}, {"<｜end▁of▁sentence｜>", v->eos_id},
        {"[gMASK]", v->bos_id}, {"<sop>", v->sop_id}, {"<|system|>", v->system_id},
        {"<｜User｜>", v->user_id}, {"<｜Assistant｜>", v->assistant_id},
        {"<|user|>", v->user_id}, {"<|assistant|>", v->assistant_id},
        {"<|observation|>", v->observation_id}, {"<think>", v->think_start_id},
        {"</think>", v->think_end_id}, {"<tool_call>", v->tool_call_start_id},
        {"</tool_call>", v->tool_call_end_id}, {"<tool_response>", v->tool_response_start_id},
        {"</tool_response>", v->tool_response_end_id}, {"<arg_key>", v->arg_key_start_id},
        {"</arg_key>", v->arg_key_end_id}, {"<arg_value>", v->arg_value_start_id},
        {"</arg_value>", v->arg_value_end_id}, {"｜DSML｜", v->dsml_id},
        {"<|im_start|>", v->im_start_id}, {"<|im_end|>", v->im_end_id},
        {"<|endoftext|>", v->endoftext_id},
        {"<｜System｜>", ds41 ? v->system_id : -1},
    };
    chat_marker mapped[CHAT_MARKER_COUNT];
    chat_markers(v, mapped);
    for (size_t w = 0; w < sizeof(want) / sizeof(want[0]); w++) {
        bool found = false;
        for (size_t i = 0; i < CHAT_MARKER_COUNT; i++) {
            if (strcmp(mapped[i].text, want[w].text)) continue;
            found = true;
            if (mapped[i].token != want[w].id) {
                fprintf(stderr, "%s: marker %s maps to %d, want %d\n", family, want[w].text,
                        mapped[i].token, want[w].id);
                exit(1);
            }
        }
        CHECK(found);
    }

    /* chat_marker_len_at: every marker, whatever bytes follow; nothing else */
    for (size_t m = 0; m < N_MARKERS; m++) {
        char text[128];
        snprintf(text, sizeof(text), "%s and more", all_markers[m]);
        CHECK(chat_marker_len_at(text) == strlen(all_markers[m]));
        CHECK(chat_marker_len_at(all_markers[m]) == strlen(all_markers[m]));
    }
    CHECK(chat_marker_len_at("｜DSML｜x") == strlen("｜DSML｜")); /* first byte 0xEF */
    CHECK(chat_marker_len_at("[gMASK]") == 7);                  /* first byte '[' */
    CHECK(chat_marker_len_at("abc") == 0 && chat_marker_len_at("") == 0);
    CHECK(chat_marker_len_at("<x>") == 0 && chat_marker_len_at("<think") == 0);
    CHECK(chat_marker_len_at("[gMASK") == 0 && chat_marker_len_at("｜DSML") == 0);
    CHECK(chat_marker_len_at("｜X｜") == 0 && chat_marker_len_at("ï") == 0);

    /* ds4_chat_escape_markers */
    check_escape_exact("", 0, "");
    check_escape_exact("abc", 3, "abc");
    check_escape_exact("a<think>b", 9, "a\x01<think>b");
    check_escape_exact("<think><think>", 14, "\x01<think>\x01<think>");
    check_escape_exact("<<|user|>", 9, "<\x01<|user|>");
    check_escape_exact("\x01", 1, "\x01\x01");
    check_escape_exact("\x01<think>", 8, "\x01\x01\x01<think>");
    check_escape_exact("｜DSML｜x", strlen("｜DSML｜x"), "\x01｜DSML｜x");
    check_escape_exact("x[gMASK]", 8, "x\x01[gMASK]");
    check_escape_exact("<think>abc", 4, "<thi");      /* len cuts a marker short */
    check_escape_exact("<think>abc", 7, "\x01<think>");
    for (size_t m = 0; m < N_MARKERS; m++) {
        char want_text[128];
        snprintf(want_text, sizeof(want_text), "%c%s", DS4_CHAT_ESCAPE, all_markers[m]);
        check_escape_exact(all_markers[m], strlen(all_markers[m]), want_text);
    }

    /* ds4_tokens_render_chat_text: specials bare, plain runs escaped as a whole */
    {
        token_vec a, b, c, all = {0};
        const int eos = v->eos_id;
        tokenize_plain(e, "x <|us", &a);
        tokenize_plain(e, "er|> y <think>", &b);
        for (int i = 0; i < a.len; i++) token_vec_push(&all, a.v[i]);
        for (int i = 0; i < b.len; i++) token_vec_push(&all, b.v[i]);
        size_t n = 0;
        char *got = ds4_tokens_render_chat_text(e, &all, &n);
        /* the marker spans two token runs here, both plain: escaped once */
        CHECK(n == strlen(got) && !strcmp(got, "x \x01<|user|> y \x01<think>"));
        free(got);

        token_vec mixed = {0};
        tokenize_plain(e, "p<|us", &c);
        for (int i = 0; i < c.len; i++) token_vec_push(&mixed, c.v[i]);
        token_vec_push(&mixed, eos);
        token_vec_free(&c);
        tokenize_plain(e, "er|>", &c);
        for (int i = 0; i < c.len; i++) token_vec_push(&mixed, c.v[i]);
        got = ds4_tokens_render_chat_text(e, &mixed, &n);
        char want_text[128];
        size_t el = 0;
        char *eos_text = ds4_token_text(e, eos, &el);
        /* a special token ends the run: the half marker before it is not a marker */
        snprintf(want_text, sizeof(want_text), "p<|us%ser|>", eos_text);
        CHECK(!strcmp(got, want_text));
        free(got);
        free(eos_text);
        token_vec_free(&mixed);
        token_vec_free(&c);

        /* the eos token renders bare even when it is not a table marker (GLM) */
        token_vec one = {0};
        token_vec_push(&one, eos);
        got = ds4_tokens_render_chat_text(e, &one, &n);
        CHECK(n > 0 && got[0] != DS4_CHAT_ESCAPE);
        free(got);
        token_vec_free(&one);
        if (!strcmp(family, "glm")) {
            tokenize_plain(e, "<|endoftext|>", &c);
            CHECK(c.len > 1); /* plain text: bytes, not the eos token */
            got = ds4_tokens_render_chat_text(e, &c, &n);
            CHECK(!strcmp(got, "\x01<|endoftext|>"));
            free(got);
            token_vec_free(&c);
        }
        token_vec_free(&a);
        token_vec_free(&b);
        token_vec_free(&all);
        got = ds4_tokens_render_chat_text(e, NULL, &n);
        CHECK(n == 0 && got[0] == 0);
        free(got);
    }

    /* ds4_chat_append_message: GLM system text is plain, not special tokens */
    if (!strcmp(family, "glm") || !strcmp(family, "deepseek4")) {
        const char *content = "x <|user|> <｜User｜> <think> [gMASK]";
        ds4_tokens t = {0}, plain;
        ds4_chat_append_message(e, &t, "system", content);
        tokenize_plain(e, content, &plain);
        int first = 0;
        if (!strcmp(family, "glm")) {
            CHECK(t.len == plain.len + 1 && t.v[0] == marker_token(e, "<|system|>"));
            first = 1;
        }
        CHECK(t.len - first == plain.len);
        CHECK(!memcmp(t.v + first, plain.v, (size_t)plain.len * sizeof(int)));
        CHECK(!has_marker_token(v, &(token_vec){t.v + first, t.len - first, t.cap}));
        ds4_tokens_free(&t);
        ds4_tokens_free(&plain);
    }
}

/* ---- disk cache keys: token-derived text against a server-shaped prompt ---- */

typedef struct {
    int types[8];
    char text[8][256];
    int n;
} log_capture;

static void capture_log(void *ud, ds4_kvstore_log_type type, const char *msg) {
    log_capture *c = ud;
    if (c->n >= 8) return;
    c->types[c->n] = (int)type;
    snprintf(c->text[c->n], sizeof(c->text[0]), "%s", msg);
    c->n++;
}

static void write_kv_stub(ds4_engine *e, const char *dir, const char *text, uint32_t tokens) {
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, strlen(text), sha);
    char path[512];
    snprintf(path, sizeof(path), "%s/%.40s.kv", dir, sha);
    FILE *fp = fopen(path, "wb");
    CHECK(fp);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER], text_len[4];
    ds4_kvstore_fill_header(h, (uint8_t)ds4_engine_model_id(e), 2, DS4_KVSTORE_REASON_COLD,
                            0, tokens, 0, 32768, 100, 100, 0);
    ds4_kvstore_le_put32(text_len, (uint32_t)strlen(text));
    CHECK(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    CHECK(fwrite(text_len, 1, 4, fp) == 4);
    CHECK(fwrite(text, 1, strlen(text), fp) == strlen(text));
    CHECK(fclose(fp) == 0);
}

static void remove_kv_dir(const char *dir) {
    char cmd[600];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dir);
    CHECK(system(cmd) == 0);
}

static void check_kvstore(ds4_engine *e, const char *family) {
    const char *pre, *post, *cut_marker;
    if (!strcmp(family, "glm")) {
        pre = "[gMASK]<sop><|user|>"; post = "<|assistant|><think>"; cut_marker = "<|assistant|>";
    } else if (!strncmp(family, "deepseek", 8)) {
        pre = "<｜begin▁of▁sentence｜><｜User｜>"; post = "<｜Assistant｜><think>";
        cut_marker = "<｜Assistant｜>";
    } else {
        pre = "<|im_start|>user\n"; post = "<|im_end|>\n<|im_start|>assistant\n<think>\n";
        cut_marker = "<|im_end|>";
    }
    const char *dirty = "see <｜User｜> <|user|> <think> \x01 <tool_call> ｜DSML｜ end";
    char *esc = escaped(dirty);
    char *prompt = concat3(pre, esc, post);
    free(esc);

    /* the server's prompt, tokenized */
    token_vec tokens;
    tokenize_rendered(e, prompt, &tokens);
    int cut_id = marker_token(e, cut_marker), cut = -1;
    for (int i = 0; i < tokens.len; i++) if (tokens.v[i] == cut_id) cut = i;
    CHECK(cut > 0);
    ds4_tokens prefix = {0};
    for (int i = 0; i < cut; i++) ds4_tokens_push(&prefix, tokens.v[i]);

    /* the key a cold snapshot gets: text from tokens == prefix of the prompt */
    size_t key_len = 0;
    char *key = ds4_kvstore_render_tokens_text(e, &prefix, &key_len);
    CHECK(key_len > 0 && key_len < strlen(prompt) && !memcmp(key, prompt, key_len));
    /* the old key: tokens as bare text, which the prompt no longer starts with */
    size_t old_len = 0;
    char *old_key = malloc(key_len * 2 + 16);
    for (int i = 0; i < prefix.len; i++) {
        size_t n = 0;
        char *piece = ds4_token_text(e, prefix.v[i], &n);
        memcpy(old_key + old_len, piece, n);
        old_len += n;
        free(piece);
    }
    old_key[old_len] = 0;
    CHECK(strcmp(old_key, key) != 0 && memcmp(old_key, prompt, old_len) != 0);

    char tmpl_a[] = "/tmp/ds4-chat-markers-kv-XXXXXX", tmpl_b[] = "/tmp/ds4-chat-markers-kv-XXXXXX";
    char *dir_a = mkdtemp(tmpl_a), *dir_b = mkdtemp(tmpl_b);
    CHECK(dir_a && dir_b);
    write_kv_stub(e, dir_a, key, 512);
    write_kv_stub(e, dir_b, old_key, 512);

    log_capture log_a = {0}, log_b = {0};
    ds4_kvstore kc_a = {0}, kc_b = {0};
    ds4_kvstore_options opt = ds4_kvstore_default_options();
    CHECK(ds4_kvstore_open(&kc_a, dir_a, 64, false, opt, "test", capture_log, &log_a));
    CHECK(ds4_kvstore_open(&kc_b, dir_b, 64, false, opt, "test", capture_log, &log_b));
    const int model_id = ds4_engine_model_id(e);

    /* a server prompt extending the cached text hits, whatever follows it */
    char *longer = concat3(prompt, " and a different question", "");
    int idx = ds4_kvstore_find_text_prefix(&kc_a, longer, model_id, 2, 32768);
    CHECK(idx >= 0 && kc_a.entry[idx].text_bytes == key_len);
    /* a snapshot keyed the old way (bare marker text) no longer matches */
    CHECK(ds4_kvstore_find_text_prefix(&kc_b, longer, model_id, 2, 32768) < 0);

    /* the lookup miss is reported at debug level; give the engine the weights
     * try_load_text wants to see and a session shell with a context size */
    ds4_tensor gate = {0};
    gate.type = DS4_TENSOR_Q4_K;
    e->weights.layer[0].ffn_gate_exps = &gate;
    ds4_session *session = calloc(1, sizeof(*session));
    CHECK(session);
    session->ctx_size = 32768;
    ds4_tokens effective = {0};
    ds4_kvstore_load_result lr;
    CHECK(ds4_kvstore_try_load_text(&kc_b, e, session, longer, &effective, &lr, NULL, false) == 0);
    int misses = 0;
    for (int i = 0; i < log_b.n; i++) {
        if (!strstr(log_b.text[i], "lookup miss")) continue;
        misses++;
        CHECK(log_b.types[i] == DS4_KVSTORE_LOG_DEBUG);
        char want_bytes[64];
        snprintf(want_bytes, sizeof(want_bytes), "prompt_bytes=%zu", strlen(longer));
        CHECK(strstr(log_b.text[i], want_bytes) != NULL);
        CHECK(strstr(log_b.text[i], "entries=1") != NULL);
    }
    CHECK(misses == 1);
    e->weights.layer[0].ffn_gate_exps = NULL;
    free(session);
    ds4_tokens_free(&effective);

    ds4_kvstore_close(&kc_a);
    ds4_kvstore_close(&kc_b);
    remove_kv_dir(dir_a);
    remove_kv_dir(dir_b);
    free(longer);
    free(old_key);
    free(key);
    free(prompt);
    ds4_tokens_free(&prefix);
    token_vec_free(&tokens);
}

static void test_synthetic(const char *label, const char *family,
                           const char *const *template_markers) {
    char *path = vocab_fixture_write(family);
    CHECK(path);
    ds4_engine *e = NULL;
    CHECK(ds4_engine_open_vocab_only(&e, path, family) == 0);
    check_family(e, label, template_markers);
    check_helpers(e, family);
    check_kvstore(e, family);
    ds4_engine_close_vocab_only(e);
    unlink(path);
    free(path);
    printf("%s: marker strings in content are plain text, template markers stay special\n", label);
}

/* Real vocabularies: the synthetic one has no merges, the real one does. */
static void test_model_file(const char *path) {
    ds4_model model;
    model_open(&model, path, false, false);
    config_validate_model(&model);
    ds4_engine *e = calloc(1, sizeof(*e));
    CHECK(e);
    vocab_load(&e->vocab, &model);
    const char *glm_turn[] = {"<|user|>", "<|assistant|>"};
    const char *ds_turn[] = {"<｜User｜>", "<｜Assistant｜>"};
    const char *qwen_turn[] = {"<|im_start|>", "<|im_end|>"};
    const char *const *turn = DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA ? glm_turn :
                              ds4_model_is_qwen4() ? qwen_turn : ds_turn;
    check_family(e, path, turn);
    vocab_free(&e->vocab);
    free(e);
    model_close(&model);
    printf("%s: OK\n", path);
}

int main(void) {
    const char *glm_turn[] = {"<|user|>", "<|assistant|>"};
    const char *ds_turn[] = {"<｜User｜>", "<｜Assistant｜>"};
    const char *qwen_turn[] = {"<|im_start|>", "<|im_end|>"};
    test_synthetic("DeepSeek V4", "deepseek4", ds_turn);
    test_synthetic("DeepSeek V4.1", "deepseek41", ds_turn);
    test_synthetic("GLM", "glm", glm_turn);
    test_synthetic("Qwen", "qwen4", qwen_turn);

    const char *models = getenv("DS4_TEST_CHAT_MARKER_MODELS");
    if (models && models[0]) {
        char *list = strdup(models);
        for (char *save = NULL, *tok = strtok_r(list, " ", &save); tok; tok = strtok_r(NULL, " ", &save))
            test_model_file(tok);
        free(list);
    } else {
        puts("real vocabularies: skipped (set DS4_TEST_CHAT_MARKER_MODELS to GGUF paths)");
    }
    puts("chat markers: OK");
    return 0;
}
