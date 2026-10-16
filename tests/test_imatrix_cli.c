/* Exercise CLI validation without opening a model or initializing a GPU. */
#define main ds4_cli_main
#include "../ds4_cli.c"
#undef main
#include <assert.h>
#include <sys/wait.h>

static void check_case(const char *const *args, int count, int expected_exit, int batch_tokens) {
    fflush(NULL);
    const pid_t pid = fork();
    assert(pid >= 0);
    if (!pid) {
        cli_config c = parse_options(count, (char **)args);
        assert(c.gen.imatrix_batch_tokens == batch_tokens);
        _exit(0);
    }
    int status;
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == expected_exit);
}

int main(void) {
    const char *values[] = {"0", "1", "8192", "8193", "-1", "oops"};
    for (size_t i = 0; i < sizeof(values)/sizeof(*values); i++) {
        const char *args[] = {"ds4", "--imatrix-dataset", "corpus", "--imatrix-out", "result",
                              "--imatrix-batch-tokens", values[i]};
        check_case(args, 7, i < 3 ? 0 : 2, i < 3 ? atoi(values[i]) : 0);
    }
    const char *missing_output[] = {"ds4", "--imatrix-batch-tokens", "8192"};
    check_case(missing_output, 3, 2, 0);
    const char *missing_checkpoint[] = {"ds4", "--imatrix-resume"};
    check_case(missing_checkpoint, 2, 2, 0);
    const char *partial_prompt[] = {"ds4", "--imatrix-dataset", "corpus", "--imatrix-out", "result",
        "--imatrix-checkpoint", "state", "--imatrix-max-tokens", "100"};
    check_case(partial_prompt, 9, 2, 0);
    const char *valid[] = {"ds4", "--imatrix-dataset", "corpus", "--imatrix-out", "result",
        "--imatrix-checkpoint", "state", "--imatrix-resume", "--imatrix-checkpoint-every", "2"};
    check_case(valid, 10, 0, 0);
    puts("Imatrix CLI zero, bounds and checkpoint validation: PASS");
    return 0;
}
