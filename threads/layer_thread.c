#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "./layer_thread.h"

void layer_thread(void* args){
    LayerThread* lt = (LayerThread*)args;

    while (lt->running){
        pthread_mutex_lock(&lt->mutex);
        for (int k = 0; k < lt->current_buffer_size; k++){
            Signal* sign;
            LayerSignal* input = lt->input_buffer[k];
            LayerSignal* output;

            output->n_signs = lt->layer->n_neurons;
            output->signs = malloc(output->n_signs*sizeof(Signal*));

            for (int i = 0; i < input->n_signs; i++){
                output->signs[i] = NULL;
                if (lt->layer->conns[input->from] == NULL){
                    for (int j = 0; j < lt->layer->n_neurons; j++){
                        Signal* inner_sign;
                        float w = lt->layer->conns[input->from][i][j];
                        inner_sign = LIF(&lt->layer->neurons[j], *input->signs[i], w, j, lt->layer->idx);
                        if (output->signs[i] == NULL){
                            output->signs[i] = inner_sign;
                            continue;
                        }
                        output->signs[i]->output += inner_sign->output;
                    }
                }
            }
        }
        pthread_mutex_unlock(&lt->mutex);
    }

}

NetworkThreads* init_layer_threads(void* args){
    Network* net = (Network*) args;
    NetworkThreads* net_ts = malloc(net->n_layers * sizeof(LayerThread*));
    for (int i = 0; i < net->n_layers; i++){
        LayerThread* lt;
        lt->running = 1;
        lt->layer = net->layers[i];
        int result = pthread_mutex_init(&lt->mutex, NULL);
        if (result > 0){
            perror("Error on init thread mutex (line 13 - layer_thread.c)");
            return;
        }

        net_ts->lts[i] = lt;
        pthread_create(&lt->thread, NULL, layer_thread, (void*)lt);
    }

    return net_ts;

}

void end_layer_threads(NetworkThreads* net_ts, Network* net){
    for (int i = 0; i < net->n_layers; i++){
        net_ts->lts[i]->running = 0;
        pthread_join(net_ts->lts[i]->thread, NULL);
    }

}
