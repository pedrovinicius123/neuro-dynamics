#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include "logger.h"

static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE* file_ptr;
static unsigned long spike_count;
static unsigned long weight_update_count;
static unsigned long spike_counts_by_layer[LOGGER_MAX_LAYERS];

void init_logger(const char* filename){
    pthread_mutex_lock(&log_mutex);
    if (file_ptr != NULL) fclose(file_ptr);
    file_ptr = filename != NULL ? fopen(filename, "ab") : NULL;
    int opened = file_ptr != NULL;
    pthread_mutex_unlock(&log_mutex);
    if (!opened){
        perror("Log finished");
        return;
    } 
    printf("Logger inited successfully\n");
}

void logger_log(Event event_type, Data data){
    pthread_mutex_lock(&log_mutex);
    if (event_type == EVENT_TYPE_SPIKE){
        spike_count++;
        if (data.layer_idx >= 0 && data.layer_idx < LOGGER_MAX_LAYERS){
            spike_counts_by_layer[data.layer_idx]++;
        }
    } else if (event_type == EVENT_TYPE_WEIGTH_UPDATE){
        weight_update_count++;
    }
    if (file_ptr == NULL) {
        pthread_mutex_unlock(&log_mutex);
        return;
    }

    /* Keep the on-disk format independent of C struct padding. */
    int event_value = (int)event_type;
    int written = 0;
    written += fwrite(&event_value, sizeof(event_value), 1, file_ptr) == 1;
    written += fwrite(&data.neuron_idx, sizeof(data.neuron_idx), 1, file_ptr) == 1;
    written += fwrite(&data.layer_idx, sizeof(data.layer_idx), 1, file_ptr) == 1;
    written += fwrite(&data.neuron_u, sizeof(data.neuron_u), 1, file_ptr) == 1;
    written += fwrite(&data.timestamp, sizeof(data.timestamp), 1, file_ptr) == 1;
    if (written == 5) fflush(file_ptr);
    
    pthread_mutex_unlock(&log_mutex);
}

void logger_reset_counts(void){
    pthread_mutex_lock(&log_mutex);
    spike_count = 0;
    weight_update_count = 0;
    for (int layer = 0; layer < LOGGER_MAX_LAYERS; layer++){
        spike_counts_by_layer[layer] = 0;
    }
    pthread_mutex_unlock(&log_mutex);
}

void logger_get_counts(unsigned long* spikes, unsigned long* weight_updates,
                       unsigned long* spikes_by_layer, size_t layer_count){
    pthread_mutex_lock(&log_mutex);
    if (spikes != NULL) *spikes = spike_count;
    if (weight_updates != NULL) *weight_updates = weight_update_count;
    if (spikes_by_layer != NULL){
        size_t copy_count = layer_count < LOGGER_MAX_LAYERS
            ? layer_count : LOGGER_MAX_LAYERS;
        for (size_t layer = 0; layer < copy_count; layer++){
            spikes_by_layer[layer] = spike_counts_by_layer[layer];
        }
    }
    pthread_mutex_unlock(&log_mutex);
}

void logger_end(){
    pthread_mutex_lock(&log_mutex);
    if (file_ptr != NULL){
        fclose(file_ptr);
        file_ptr = NULL;
    }
    pthread_mutex_unlock(&log_mutex);
}
