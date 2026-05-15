#define SNN_LIF_PROC
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "./layer_thread.h"

void signal(NetworkThreads* ntts, LayerSignal* ls){
    for (int nl = 0; nl < ntts->n_layers; nl++){
        if (nl != ls->from && ntts->layers[nl]->conns[ls->from] != NULL){
            

        }
    }
}

void* thread(void* args){
    ThreadEntry* te = (ThreadEntry*) args;
    while (te->lt->running){    
        for(int buffer = 0; buffer < te->lt->current_buffer_size; buffer++){
            pthread_mutex_lock(&te->lt->mutex);
            LayerSignal* sign = te->lt->input_buffer[buffer];
            printf("Signal Going! %d\n", te->lt->layer->idx);
            signal(te->net, sign);
            pthread_mutex_unlock(&te->lt->mutex);
        }    
    }

    return NULL;
}


NetworkThreads* init_layer_threads(Network* net){
    NetworkThreads* ntts = malloc(sizeof(NetworkThreads));
    ntts->n_layers = net->n_layers;
    for (int nl = 0; nl < net->n_layers; nl++){
        LayerThread* lt = ntts->lts[nl];
        lt->layer = net->layers[nl];
        lt->running = 1;
        lt->current_buffer_size = 0;

        pthread_mutex_init(&lt->thread, NULL);
        pthread_cond_init(&lt->cond, NULL);
        pthread_create(&lt->thread, NULL, thread, (void*)lt);
    }

    return ntts;

}

void end_layer_threads(NetworkThreads* ntts){
    for(int nl = 0; nl < ntts->n_layers; nl++){
        LayerThread* lt = ntts->lts[nl];
        
        lt->running = 0;
        pthread_join(lt->thread, NULL);
        pthread_mutex_destroy(&lt->mutex);
    }

    printf("Threads succesfully stoped\n");

}
