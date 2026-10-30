#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../neura.c"

static void test_transpose(void){
    float row0[] = {1.0f, 2.0f, 3.0f};
    float row1[] = {4.0f, 5.0f, 6.0f};
    float* matrix[] = {row0, row1};
    float** result = transpose(matrix, 2, 3);
    assert(result != NULL);
    assert(result[0][0] == 1.0f && result[0][1] == 4.0f);
    assert(result[2][0] == 3.0f && result[2][1] == 6.0f);
    for (int row = 0; row < 3; row++) free(result[row]);
    free(result);
}

static void test_architecture_round_trip(void){
    Network* generated = read_neural_architecture("neura_architectures/n1.neur");
    assert(generated != NULL);
    assert(generated->n_layers == 6);
    assert(generated->layers[0]->n_neurons == 10);
    assert(generated->layers[0]->label != NULL);
    assert(generated->layers[0]->conns[1] != NULL);

    NetworkThreads* threads = init_layer_threads(generated);
    assert(threads != NULL);
    end_layer_threads(threads);

    const char* filename = "/tmp/neuro-dynamics-test.model";
    write_neural_bins(filename, generated);
    free_network(generated);

    Network loaded = read_neural_bins(filename);
    assert(loaded.n_layers == 6);
    assert(loaded.layers[5]->n_neurons == 20);
    assert(loaded.layers[0]->conns[1] != NULL);
    assert(isfinite(loaded.layers[0]->conns[1][0][0]));
    free_network_contents(&loaded);
    remove(filename);
}

static void test_rejects_invalid_model(void){
    const char* filename = "/tmp/neuro-dynamics-invalid.model";
    FILE* file = fopen(filename, "wb");
    assert(file != NULL);
    int invalid_layer_count = MAX_LAYERS + 1;
    assert(fwrite(&invalid_layer_count, sizeof(int), 1, file) == 1);
    fclose(file);

    Network loaded = read_neural_bins(filename);
    assert(loaded.n_layers == 0);
    remove(filename);
}

int main(void){
    test_transpose();
    test_architecture_round_trip();
    test_rejects_invalid_model();
    puts("C tests passed");
    return 0;
}
