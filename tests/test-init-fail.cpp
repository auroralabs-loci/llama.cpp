// Regression test for: sampler() segfault when model fails to load
// When common_init_from_params fails to load a model, the samplers vector
// is empty. Calling sampler() without bounds checking caused a segfault
// due to out-of-bounds access on the empty vector.

#include "common.h"
#include "llama.h"

#include <cstdio>

#undef NDEBUG
#include <cassert>

int main(void) {
    common_init();

    common_params params;
    params.model.path = "/non_existent_model.gguf";

    auto result = common_init_from_params(params);

    assert(result->model()    == nullptr);
    assert(result->context()  == nullptr);
    assert(result->sampler(0) == nullptr);

    printf("%s: OK\n", __func__);
    return 0;
}
