/* Small, model-free round-trip test for the routed-imatrix checkpoint format. */
#include <assert.h>
#include "../ds4.c"

int main(void) {
    const ds4_shape original = g_ds4_shape;
    g_ds4_shape.n_layer = 2;
    g_ds4_shape.n_expert = 3;
    g_ds4_shape.n_expert_used = 2;
    g_ds4_shape.n_embd = 4;
    g_ds4_shape.n_ff_exp = 5;

    char directory[] = "/tmp/ds4-imatrix-ckpt-test.XXXXXX";
    assert(mkdtemp(directory) != NULL);
    char checkpoint_path[256], output_path[256];
    snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/state.ckpt", directory);
    snprintf(output_path, sizeof(output_path), "%s/result.dat", directory);
    const char *dataset =
        "===== DS4_IMATRIX_PROMPT first =====\nhello\n"
        "===== DS4_IMATRIX_PROMPT second =====\nworld\n";
    const char *second = strstr(dataset + 1, "===== DS4_IMATRIX_PROMPT");
    assert(second != NULL);

    ds4_imatrix_collector first, restored;
    assert(imatrix_collector_init(&first, 8, "synthetic-dataset"));
    assert(imatrix_collector_init(&restored, 8, "synthetic-dataset"));
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        for (uint32_t expert = 0; expert < DS4_N_EXPERT; expert++) {
            const uint32_t count = il ? expert + 1 : 3 - expert;
            first.gate_up_count[il][expert] = count;
            first.down_count[il][expert] = count;
            for (uint32_t i = 0; i < DS4_N_EMBD; i++)
                imatrix_gate_up_ptr(&first, il, expert)[i] =
                    (float)(100 * il + 10 * expert + i);
            for (uint32_t i = 0; i < DS4_N_FF_EXP; i++)
                imatrix_down_ptr(&first, il, expert)[i] =
                    (float)(200 * il + 20 * expert + i);
        }
    }
    first.observed_tokens = 6;
    first.observed_routes = 12;
    first.chunks = 4;

    ds4_imatrix_checkpoint writer, reader;
    assert(imatrix_checkpoint_prepare(&writer, checkpoint_path, false,
            "tests/test_imatrix_checkpoint.c", "synthetic-dataset", output_path,
            dataset, strlen(dataset), &first, 2, 128));
    assert(imatrix_checkpoint_save(&writer, &first,
            (size_t)(second - dataset), 2, 3));
    assert(imatrix_checkpoint_prepare(&reader, checkpoint_path, true,
            "tests/test_imatrix_checkpoint.c", "synthetic-dataset", output_path,
            dataset, strlen(dataset), &restored, 2, 128));
    size_t offset = 0;
    int prompts = 0, tokens = 0;
    assert(imatrix_checkpoint_load(&reader, &restored, dataset, strlen(dataset),
            &offset, &prompts, &tokens));
    assert(offset == (size_t)(second - dataset) && prompts == 2 && tokens == 3);
    assert(restored.observed_tokens == 6 && restored.observed_routes == 12 &&
           restored.chunks == 4);
    assert(memcmp(first.gate_up_sum2, restored.gate_up_sum2,
            (size_t)writer.expected[IMCK_GATE_BYTES]) == 0);
    assert(memcmp(first.down_sum2, restored.down_sum2,
            (size_t)writer.expected[IMCK_DOWN_BYTES]) == 0);
    assert(memcmp(first.gate_up_count, restored.gate_up_count,
            sizeof(first.gate_up_count)) == 0);
    assert(memcmp(first.down_count, restored.down_count,
            sizeof(first.down_count)) == 0);

    /* A later atomic save replaces the previous checkpoint. */
    first.gate_up_sum2[0] += 7.0f;
    first.down_sum2[0] += 11.0f;
    first.chunks++;
    assert(imatrix_checkpoint_save(&writer, &first,
            (size_t)(second - dataset), 2, 3));
    assert(imatrix_checkpoint_load(&reader, &restored, dataset, strlen(dataset),
            &offset, &prompts, &tokens));
    assert(restored.gate_up_sum2[0] == first.gate_up_sum2[0] &&
           restored.down_sum2[0] == first.down_sum2[0] && restored.chunks == 5);

    /* Existing state cannot be overwritten by a new run without --resume. */
    ds4_imatrix_checkpoint rejected;
    assert(!imatrix_checkpoint_prepare(&rejected, checkpoint_path, false,
            "tests/test_imatrix_checkpoint.c", "synthetic-dataset", output_path,
            dataset, strlen(dataset), &first, 2, 128));
    assert(imatrix_checkpoint_prepare(&rejected, checkpoint_path, true,
            "tests/test_imatrix_checkpoint.c", "synthetic-dataset", output_path,
            "changed dataset", strlen("changed dataset"), &first, 2, 128));
    assert(!imatrix_checkpoint_load(&rejected, &restored, dataset, strlen(dataset),
            &offset, &prompts, &tokens));

    /* Every identity/geometry field is checked, including device IDs which
     * can change across an OS upgrade or a remount. Do not silently bypass it. */
    for (int field = 0; field < IMCK_CURSOR; field++) {
        rejected = reader;
        rejected.expected[field]++;
        assert(!imatrix_checkpoint_load(&rejected, &restored, dataset, strlen(dataset),
                &offset, &prompts, &tokens));
    }
    assert(imatrix_checkpoint_prepare(&rejected, checkpoint_path, true,
            "ds4.c", "synthetic-dataset", output_path,
            dataset, strlen(dataset), &first, 2, 128));
    assert(!imatrix_checkpoint_load(&rejected, &restored, dataset, strlen(dataset),
            &offset, &prompts, &tokens));

    FILE *fp = fopen(checkpoint_path, "r+b");
    assert(fp != NULL);
    assert(fseek(fp, 8 + (long)sizeof(writer.expected) + 3, SEEK_SET) == 0);
    const int byte = fgetc(fp);
    assert(byte != EOF && fseek(fp, -1, SEEK_CUR) == 0);
    assert(fputc(byte ^ 1, fp) != EOF && fclose(fp) == 0);
    assert(!imatrix_checkpoint_load(&reader, &restored, dataset, strlen(dataset),
            &offset, &prompts, &tokens));

    assert(imatrix_checkpoint_save(&writer, &first, strlen(dataset), 2, 3));
    assert(imatrix_checkpoint_load(&reader, &restored, dataset, strlen(dataset),
            &offset, &prompts, &tokens));
    assert(offset == strlen(dataset));
    struct stat st;
    assert(stat(checkpoint_path, &st) == 0 && truncate(checkpoint_path, st.st_size - 1) == 0);
    assert(!imatrix_checkpoint_load(&reader, &restored, dataset, strlen(dataset),
            &offset, &prompts, &tokens));

    assert(unlink(checkpoint_path) == 0);
    assert(rmdir(directory) == 0);
    imatrix_collector_free(&first);
    imatrix_collector_free(&restored);
    g_ds4_shape = original;
    puts("imatrix checkpoint round-trip and rejection tests passed");
    return 0;
}
