#define SNN_LIF_PROC
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "./layer_thread.h"

static void propagate_signal(Network* net, LayerSignal* input, int lif_only,
                             int learn, int* visited){
    if (net == NULL || input == NULL || input->outputs == NULL) return;
    if (input->from < 0 || input->from >= net->n_layers) return;

    LayerSignal signals[MAX_LAYERS] = {0};
    int queue[MAX_LAYERS];
    int queue_start = 0;
    int queue_end = 0;
    int initial_source = input->from;
    if (input->n_outputs > net->layers[initial_source]->n_neurons){
        input->n_outputs = net->layers[initial_source]->n_neurons;
    }
    signals[initial_source] = *input;
    visited[initial_source] = 1;
    queue[queue_end++] = initial_source;

    while (queue_start < queue_end){
        int source = queue[queue_start++];
        LayerSignal* source_signal = &signals[source];
        for (int target_index = 0; target_index < net->n_layers; target_index++){
            Layer* target = net->layers[target_index];
            if (target == NULL || target_index == source || visited[target_index]
                || target->conns[source] == NULL) continue;

            LayerSignal output = LIF_Signal(*source_signal, lif_only, target,
                                             target->n_neurons, target->conns);
            if (output.outputs == NULL || output.spike_timestamps == NULL){
                free(output.outputs);
                free(output.spike_timestamps);
                continue;
            }
            if (learn) STDP(*source_signal, target);
            signals[target_index] = output;
            visited[target_index] = 1;
            queue[queue_end++] = target_index;
        }
    }

    for (int layer = 0; layer < net->n_layers; layer++){
        if (layer == initial_source) continue;
        free(signals[layer].outputs);
        free(signals[layer].spike_timestamps);
    }
}

static void signal_internal(Network* net, LayerSignal* s, int lif_only, int learn){
    if (net == NULL || s == NULL || net->n_layers <= 0) return;

    /* External sources (audio, tokenizers) enter through the input layer. */
    LayerSignal input = *s;
    if (input.from < 0 || input.from >= net->n_layers) input.from = 0;

    int visited[MAX_LAYERS] = {0};
    propagate_signal(net, &input, lif_only, learn, visited);
}

void signal(Network* net, LayerSignal* s, int lif_only){
    signal_internal(net, s, lif_only, 1);
}

void signal_inference(Network* net, LayerSignal* s, int lif_only){
    signal_internal(net, s, lif_only, 0);
}

void* thread(void* args){
    LayerThread* lt = (LayerThread*)args;
    for (;;){
        pthread_mutex_lock(&lt->mutex);
        while (lt->current_buffer_size == 0 && lt->running){
            pthread_cond_wait(&lt->cond, &lt->mutex);
        }
        if (lt->current_buffer_size == 0 && !lt->running){
            pthread_mutex_unlock(&lt->mutex);
            break;
        }
        LayerSignal* ls = lt->input_buffer[0];
        for (int buffer = 1; buffer < lt->current_buffer_size; buffer++){
            lt->input_buffer[buffer - 1] = lt->input_buffer[buffer];
        }
        lt->current_buffer_size--;
        lt->input_buffer[lt->current_buffer_size] = NULL;
        pthread_mutex_unlock(&lt->mutex);

        if (ls != NULL){
            free(ls->outputs);
            free(ls->spike_timestamps);
            free(ls);
        }
    }
    return NULL;
}

NetworkThreads* init_layer_threads(Network* net){
    if (net == NULL || net->n_layers <= 0 || net->n_layers > MAX_LAYERS) return NULL;
    printf("NLAYERS %d\n", net->n_layers);
    NetworkThreads* net_ts = calloc(1, sizeof(NetworkThreads));
    if (net_ts == NULL) return NULL;
    net_ts->lts = calloc((size_t)net->n_layers, sizeof(LayerThread*));
    if (net_ts->lts == NULL){
        free(net_ts);
        return NULL;
    }

    for (int i = 0; i < net->n_layers; i++){
        LayerThread* lt = calloc(1, sizeof(LayerThread));
        if (lt == NULL){
            end_layer_threads(net_ts);
            return NULL;
        }
        lt->layer = net->layers[i];
        lt->running = 1;
        if (pthread_mutex_init(&lt->mutex, NULL) != 0){
            free(lt);
            end_layer_threads(net_ts);
            return NULL;
        }
        if (pthread_cond_init(&lt->cond, NULL) != 0){
            pthread_mutex_destroy(&lt->mutex);
            free(lt);
            end_layer_threads(net_ts);
            return NULL;
        }
        if (pthread_create(&lt->thread, NULL, thread, (void*)lt) != 0){
            pthread_cond_destroy(&lt->cond);
            pthread_mutex_destroy(&lt->mutex);
            free(lt);
            end_layer_threads(net_ts);
            return NULL;
        }

        net_ts->lts[i] = lt;
        net_ts->n_layers++;
    }
    return net_ts;

}

void end_layer_threads(NetworkThreads* netts){
    if (netts == NULL) return;
    for (int i = 0; i < netts->n_layers; i++){
        LayerThread* lt = netts->lts[i];
        if (lt == NULL) continue;
        pthread_mutex_lock(&lt->mutex);
        lt->running = 0;
        pthread_cond_broadcast(&lt->cond);
        pthread_mutex_unlock(&lt->mutex);
        pthread_join(lt->thread, NULL);
        pthread_mutex_destroy(&lt->mutex);
        pthread_cond_destroy(&lt->cond);
        free(lt);
    }
    free(netts->lts);
    free(netts);
    printf("Layer threads uninited successfully\n");
}
