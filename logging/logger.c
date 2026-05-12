#include <stdlib.h>
#include <stdio.h>
#include "logger.h"

FILE* file_ptr;

void init_logger(const char* filename){
    file_ptr = fopen(filename, "wb");    
}

void logger_log(Event event_type, Data data){
    LogEntry* log;
    log->event_type = event_type;
    log->data = data;
    fwrite(log, sizeof(LogEntry), 1, file_ptr);
}
