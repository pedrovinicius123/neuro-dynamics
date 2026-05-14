#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include "logger.h"

static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
FILE* file_ptr;

void init_logger(const char* filename){
    file_ptr = fopen(filename, "ab"); 
    if (!file_ptr){
        perror("Log finished");
        return;
    } 
    printf("Logger inited successfully\n");
}

void logger_log(Event event_type, Data data){
    pthread_mutex_lock(&log_mutex);
    
    LogEntry log;
    log.event_type = event_type;
    log.data = data;
    fwrite(&log, sizeof(LogEntry), 1, file_ptr);
    fflush(file_ptr);
    
    pthread_mutex_unlock(&log_mutex);
    printf("Logging successful\n");
}

void logger_end(){
    fclose(file_ptr);
    printf("Logger closed\n");
}
