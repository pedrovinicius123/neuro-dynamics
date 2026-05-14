#define SNN_LIF_PROC
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "./layer_thread.h"

NetworkThreads* net_ts;
void signal(LayerSignal s){
    // Protege acesso ao buffer com mutex
    for (int i = 0; i < net_ts->n_layers; i++){
        LayerThread* lt = net_ts->lts[i];
        if (!lt || !lt->running) continue;
        
        int should_process = 0;
        
        if (s.from == MAX_LAYERS){
            // Sinal de entrada - procura camada "INPUT"
            if (lt->layer->label && strcmp(lt->layer->label, "INPUT") == 0){
                should_process = 1;
            }
        } else {
            // Verifica se esta camada tem conexão com a camada de origem
            if (s.from < MAX_LAYERS && lt->layer->conns[s.from] != NULL){
                should_process = 1;
            }
        }
        
        if (should_process){
            pthread_mutex_lock(&lt->mutex);
            
            // Verifica se há espaço no buffer
            if (lt->current_buffer_size >= MAX_BUFFER_SIZE){
                fprintf(stderr, "Buffer overflow for layer %s!\n", 
                        lt->layer->label ? lt->layer->label : "unknown");
                pthread_mutex_unlock(&lt->mutex);
                continue;
            }
            
            // ALOCA uma cópia do sinal (CORRIGIDO!)
            LayerSignal* s_copy = (LayerSignal*)malloc(sizeof(LayerSignal));
            if (!s_copy){
                fprintf(stderr, "Failed to allocate signal copy\n");
                pthread_mutex_unlock(&lt->mutex);
                continue;
            }
            
            // Copia os dados
            memcpy(s_copy, &s, sizeof(LayerSignal));
            
            // Se outputs e spike_timestamps foram alocados dinamicamente,
            // precisamos copiá-los também
            if (s.outputs){
                s_copy->outputs = (float*)malloc(s.n_outputs * sizeof(float));
                if (s_copy->outputs){
                    memcpy(s_copy->outputs, s.outputs, s.n_outputs * sizeof(float));
                }
            }
            if (s.spike_timestamps){
                s_copy->spike_timestamps = (float*)malloc(s.n_outputs * sizeof(float));
                if (s_copy->spike_timestamps){
                    memcpy(s_copy->spike_timestamps, s.spike_timestamps, 
                           s.n_outputs * sizeof(float));
                }
            }
            
            // Adiciona ao buffer
            lt->input_buffer[lt->current_buffer_size] = s_copy;
            lt->current_buffer_size++;  // CORRIGIDO: incrementa o valor
            
            // Sinaliza a thread que há dados disponíveis
            pthread_cond_signal(&lt->cond);
            pthread_mutex_unlock(&lt->mutex);
        }
    }
}

void* layer_thread(void* args){
    LayerThread* lt = (LayerThread*)args;
    
    // Verifica se lt é válido
    if (!lt || !lt->layer) {
        fprintf(stderr, "Invalid LayerThread\n");
        return NULL;
    }
    
    printf("Thread started for layer %s\n", lt->layer->label ? lt->layer->label : "unknown");
    
    
    while (lt->running){ 
        pthread_mutex_lock(&lt->mutex);      
        // Espera por trabalho (evita busy waiting)
        while (lt->current_buffer_size == 0 && lt->running) {
            pthread_cond_wait(&lt->cond, &lt->mutex);
        }
        
        if (!lt->running) {
            pthread_mutex_unlock(&lt->mutex);
            break;
        }
        
        // Processa buffer
        for (int k = 0; k < lt->current_buffer_size; k++){
            LayerSignal* input = lt->input_buffer[k];
            if (!input) continue;
            
            int lif_only = 0;
            if (input->from == MAX_LAYERS){
                lif_only = 1;
            }
            
            if (!input->outputs || !input->spike_timestamps) {
                fprintf(stderr, "Failed to allocate outputs\n");
                free(input);
                continue;
            }

            LayerSignal output = LIF_Signal(*input, lif_only, lt->layer, 
                                           lt->layer->n_neurons, lt->layer->conns);
            if (!lif_only){
                STDP(*input, lt->layer);
            }
            signal(output);
            
            free(input->outputs);
            free(input->spike_timestamps);
            free(input);
            lt->input_buffer[k] = NULL;
            lt->current_buffer_size--;
        }
        pthread_mutex_unlock(&lt->mutex);
    }
    
    
    printf("Thread ending for layer %s\n", lt->layer->label ? lt->layer->label : "unknown");
    return NULL;
}

void init_layer_threads(void* args){
    Network* net = (Network*) args;
    
    // Aloca net_ts
    net_ts = (NetworkThreads*)malloc(sizeof(NetworkThreads));
    if (!net_ts) {
        fprintf(stderr, "Failed to allocate NetworkThreads\n");
        return;
    }
    
    // Inicializa net_ts
    net_ts->n_layers = net->n_layers;
    net_ts->lts = (LayerThread**)malloc(net->n_layers * sizeof(LayerThread*));
    if (!net_ts->lts) {
        fprintf(stderr, "Failed to allocate lts array\n");
        free(net_ts);
        net_ts = NULL;
        return;
    }
    
    for (int i = 0; i < net->n_layers; i++){
        printf("Begin layer %d!\n", i);

        LayerThread* lt = (LayerThread*)malloc(sizeof(LayerThread));
        if (!lt) {
            fprintf(stderr, "Failed to allocate LayerThread %d\n", i);
            // Limpa alocações anteriores
            for (int j = 0; j < i; j++) {
                free(net_ts->lts[j]);
            }
            free(net_ts->lts);
            free(net_ts);
            net_ts = NULL;
            return;
        }
        
        // Inicializa TODOS os campos
        memset(lt, 0, sizeof(LayerThread));  // Zera tudo primeiro
        lt->running = 1;
        lt->layer = net->layers[i];
        lt->current_buffer_size = 0;
                
        printf("Creating layer %d\n", i);
        
        // Inicializa mutex
        int result = pthread_mutex_init(&lt->mutex, NULL);
        if (result != 0){
            fprintf(stderr, "Error on init thread mutex: %s\n", strerror(result));
            free(lt);
            // Limpa alocações anteriores
            for (int j = 0; j < i; j++) {
                pthread_mutex_destroy(&net_ts->lts[j]->mutex);
                free(net_ts->lts[j]);
            }
            free(net_ts->lts);
            free(net_ts);
            net_ts = NULL;
            return;
        }
        
        // Inicializa variável de condição (se necessário)
        result = pthread_cond_init(&lt->cond, NULL);
        if (result != 0){
            fprintf(stderr, "Error on init thread cond: %s\n", strerror(result));
            pthread_mutex_destroy(&lt->mutex);
            free(lt);
            // Limpa...
            return;
        }
        
        // Armazena ANTES de criar a thread
        net_ts->lts[i] = lt;
        
        printf("BEFORE CRASH\n");
        // Cria a thread
        result = pthread_create(&lt->thread, NULL, (void*)layer_thread, (void*)lt);
        if (result != 0){
            fprintf(stderr, "Error creating thread %d: %s\n", i, strerror(result));
            pthread_cond_destroy(&lt->cond);
            pthread_mutex_destroy(&lt->mutex);
            free(lt);
            net_ts->lts[i] = NULL;
            // Limpa...
            return;
        }
        printf("AFTER CRASH\n");
    }

    printf("Threads created successfully\n");
}

void end_layer_threads(){
    for (int i = 0; i < net_ts->n_layers; i++){
        if (net_ts->lts[i] == NULL) continue;

        net_ts->lts[i]->running = 0;
        pthread_cond_signal(&net_ts->lts[i]->cond);
        pthread_join(net_ts->lts[i]->thread, NULL);

        pthread_mutex_destroy(&net_ts->lts[i]->mutex);
        pthread_cond_destroy(&net_ts->lts[i]->cond);
       
    }

}
