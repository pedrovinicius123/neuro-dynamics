#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include "logger.h"

static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE* file_ptr;

void init_logger(const char* filename){
    if (file_ptr != NULL) logger_end();
    file_ptr = fopen(filename, "ab"); 
    if (!file_ptr){
        perror("Log finished");
        return;
    } 
    printf("Logger inited successfully\n");
}

void logger_log(Event event_type, Data data){
    pthread_mutex_lock(&log_mutex);
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

void logger_end(){
    if (file_ptr == NULL) return;
    pthread_mutex_lock(&log_mutex);
    fclose(file_ptr);
    file_ptr = NULL;
    pthread_mutex_unlock(&log_mutex);
}
