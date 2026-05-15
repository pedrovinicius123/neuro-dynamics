#ifndef ASTREAM
#define ASTREAM
#define AUDIO_BUFFER_MAX_SIZE 100
#include "../threads/layer_thread.h"
#include "miniaudio.h"

typedef struct {
    float input_buffer[3][AUDIO_BUFFER_MAX_SIZE];
    int running;
    int input_buffer_current_size;
    int read_pos;   // ADICIONADO: posição de leitura
    int write_pos;  // ADICIONADO: posição de escrita
    int label_true;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} AudioStreamThread;

typedef struct {
    Network* net;
    AudioStreamThread* asth;
} StreamEntry;

void init_audio(Network* net, ma_device* device, int label_true);
void end_audio(ma_device* device);

#endif
