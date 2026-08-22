#define SNN_LIF_PROC
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "./layer_thread.h"

static void propagate_signal(Network* net, LayerSignal* input, int lif_only,
                             int* visited){
    if (net == NULL || input == NULL || input->outputs == NULL) return;
    if (input->from < 0 || input->from >= net->n_layers) return;

    int source = input->from;
    if (input->n_outputs > net->layers[source]->n_neurons) {
        input->n_outputs = net->layers[source]->n_neurons;
    }
    visited[source] = 1;
    for (int n = 0; n < net->n_layers; n++){
        Layer* target = net->layers[n];
        if (target == NULL || target->idx == source || visited[target->idx]) continue;
        if (target->conns[source] == NULL) continue;

        LayerSignal output = LIF_Signal(*input, lif_only, target,
                                         target->n_neurons, target->conns);
        STDP(*input, target);
        visited[target->idx] = 1;
        propagate_signal(net, &output, lif_only, visited);
        free(output.outputs);
        free(output.spike_timestamps);
    }
}

void signal(Network* net, LayerSignal* s, int lif_only){
    if (net == NULL || s == NULL || net->n_layers <= 0) return;

    /* External sources (audio, tokenizers) enter through the input layer. */
    LayerSignal input = *s;
    if (input.from < 0 || input.from >= net->n_layers) input.from = 0;

    int visited[MAX_LAYERS] = {0};
    propagate_signal(net, &input, lif_only, visited);
}

void* thread(void* args){
    LayerThread* lt = (LayerThread*)args;
    while (lt->running){
        for (int buffer = 0; buffer < lt->current_buffer_size; buffer++){
            pthread_mutex_lock(&lt->mutex);
            LayerSignal* ls = lt->input_buffer[buffer];
            free(ls->outputs);
            free(ls->spike_timestamps);
            free(ls);
            printf("INPUT BUFFER IN %d\n", lt->layer->idx);
            lt->current_buffer_size--;
            pthread_mutex_unlock(&lt->mutex);
        }
    }
    return NULL;
}

NetworkThreads* init_layer_threads(Network* net){
    printf("NLAYERS %d\n", net->n_layers);
    NetworkThreads* net_ts = malloc(sizeof(NetworkThreads));
    if (net_ts == NULL) return NULL;
    net_ts->n_layers = net->n_layers;
    net_ts->lts = calloc((size_t)net->n_layers, sizeof(LayerThread*));
    if (net_ts->lts == NULL){
        free(net_ts);
        return NULL;
    }

    for (int i = 0; i < net->n_layers; i++){
        LayerThread* lt = malloc(sizeof(LayerThread));
        lt->layer = net->layers[i];
        lt->running = 1;
        lt->current_buffer_size = 0;

        pthread_mutex_init(&lt->mutex, NULL);
        pthread_cond_init(&lt->cond, NULL);
        pthread_create(&lt->thread, NULL, thread, (void*)lt);

        net_ts->lts[i] = lt;
    }
    return net_ts;

}

void end_layer_threads(NetworkThreads* netts){
    for (int i = 0; i < netts->n_layers; i++){
        netts->lts[i]->running = 0;
        pthread_join(netts->lts[i]->thread, NULL);
        pthread_mutex_destroy(&netts->lts[i]->mutex);
        pthread_cond_destroy(&netts->lts[i]->cond);
        free(netts->lts[i]);
    }
    free(netts->lts);
    free(netts);
    printf("Layer threads uninited successfully\n");
}
