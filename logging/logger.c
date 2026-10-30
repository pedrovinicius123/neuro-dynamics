#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* log_file_path = NULL;

void set_log_file(const char* path){
    log_file_path = (char*)malloc(strlen(path) + 1);
    if (log_file_path != NULL){
        strcpy(log_file_path, path);
    }
}

void log_message(const char* message, ...){
    FILE* log_file_ptr = fopen(log_file_path, "a");
    if (log_file_ptr == NULL){
        fprintf(stderr, "Failed to open log file.\n");
        return;
    }
    fprintf(log_file_ptr, "%s\n", message);
    fclose(log_file_ptr);
}

void log_event(const char* event_type, const char* event_details){
    FILE* log_file_ptr = fopen(log_file_path, "a");
    if (log_file_ptr == NULL){
        fprintf(stderr, "Failed to open log file.\n");
        return;
    }
    fprintf(log_file_ptr, "[%s] %s\n", event_type, event_details);
    fclose(log_file_ptr);
}

void log_error(const char* message){
    FILE* log_file_ptr = fopen(log_file_path, "a");
    if (log_file_ptr == NULL){
        fprintf(stderr, "Failed to open log file.\n");
        return;
    }
    fprintf(log_file_ptr, "[ERROR] %s\n", message);
    fclose(log_file_ptr);
}

void log_info(const char* message){
    FILE* log_file_ptr = fopen(log_file_path, "a");
    if (log_file_ptr == NULL){
        fprintf(stderr, "Failed to open log file.\n");
        return;
    }
    fprintf(log_file_ptr, "[INFO] %s\n", message);
    fclose(log_file_ptr);
}

void log_warning(const char* message){
    FILE* log_file_ptr = fopen(log_file_path, "a");
    if (log_file_ptr == NULL){
        fprintf(stderr, "Failed to open log file.\n");
        return;
    }
    fprintf(log_file_ptr, "[WARNING] %s\n", message);
    fclose(log_file_ptr);
}
