#ifndef LAYER_THREAD
#define LAYER_THREAD
#define MAX_BUFFER_SIZE 100
#include "../snn_lif_stdp.h"
#include <pthread.h>

typedef struct {
    Layer* layer;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t thread;

    LayerSignal* input_buffer[MAX_BUFFER_SIZE];
    int current_buffer_size;
    int running;

} LayerThread;

typedef struct {
    LayerThread** lts;
    int n_layers;
} NetworkThreads;


void signal(Network* net, LayerSignal* s, int lif_only);
NetworkThreads* init_layer_threads(Network* net);
void end_layer_threads(NetworkThreads* ntts);

#endif
