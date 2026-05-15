#ifndef LAYER_THREAD
#define LAYER_THREAD
#define MAX_BUFFER_SIZE 100
#include "../snn_lif_stdp.h"
#include <pthread.h>

typedef struct {
    Layer* layer;
    pthread_mutex_t mutex;
    pthread_t thread;
    pthread_cond_t cond;

    LayerSignal* input_buffer[MAX_BUFFER_SIZE];
    int current_buffer_size;
    int running;

} LayerThread;

typedef struct {
    LayerThread** lts;
    int n_layers;
} NetworkThreads;

typedef struct {
    Network* net;
    LayerThread* lt;
} ThreadEntry;

void signal(LayerSignal s);
void init_layer_threads(Network* net);
void end_layer_threads();

#endif
