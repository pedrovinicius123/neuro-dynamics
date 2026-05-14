#ifndef LOGGER
#define LOGGER

typedef enum {
    EVENT_TYPE_SPIKE,
    EVENT_TYPE_WEIGTH_UPDATE
} Event;

typedef struct {
    int neuron_idx;
    int layer_idx;
    float neuron_u;
    float timestamp;

} Data;

typedef struct {
    Event event_type;
    Data data;
    
} LogEntry;

void init_logger(const char* filename);
void logger_log(Event event_type, Data data);
void logger_end();

#endif
