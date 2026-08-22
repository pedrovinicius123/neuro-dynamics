#define MINIAUDIO_IMPLEMENTATION
#define MA_DEBUG_OUTPUT
#define AUDIO_BUFFER_MAX_SIZE 100
#include "astream.h"
#include "../snn_lif_stdp.h"
#include "../threads/layer_thread.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float left;
    float right;
} audio_frame_t;

AudioStreamThread* asth;
audio_frame_t temp_buffer[1024];
int temp_frames = 0;

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount){
    (void)pOutput;
    if (pInput == NULL) return;
    
    const float* pInputFloat = (const float*) pInput;
    int channels = pDevice->capture.channels;
    if (frameCount > 1024) frameCount = 1024;
    
    // COPIA RÁPIDA para buffer local (sem lock)
    for(ma_uint32 frame = 0; frame < frameCount; frame++) {
        int base_idx = frame * channels;
        temp_buffer[frame].left = pInputFloat[base_idx];
        temp_buffer[frame].right = (channels > 1) ? 
            pInputFloat[base_idx + 1] : 
            pInputFloat[base_idx];
    }
    temp_frames = frameCount;
    
    // LOCK MÍNIMO - apenas para transferir o buffer
    pthread_mutex_lock(&asth->mutex);
    
    // Transfere do temp_buffer para o ringbuffer (ainda faz loop, mas lock menor)
    for(int frame = 0; frame < temp_frames; frame++) {
        int next_write = (asth->write_pos + 1) % AUDIO_BUFFER_MAX_SIZE;
        if (next_write == asth->read_pos) {
            asth->read_pos = (asth->read_pos + 1) % AUDIO_BUFFER_MAX_SIZE;
        }
        
        asth->input_buffer[0][asth->write_pos] = temp_buffer[frame].left;
        asth->input_buffer[1][asth->write_pos] = temp_buffer[frame].right;
        asth->write_pos = (asth->write_pos + 1) % AUDIO_BUFFER_MAX_SIZE;
        asth->input_buffer_current_size++;
    }
    
    pthread_cond_signal(&asth->cond);
    pthread_mutex_unlock(&asth->mutex);
}

// Thread consumidora CORRIGIDA
void* audio_stream_thread(void* args){
    StreamEntry* ent = (StreamEntry*)args;
    float dt = 0.0f;  // MOVED para fora do loop
    while(1){        
        pthread_mutex_lock(&ent->asth->mutex);
        // Espera enquanto não há dados
        while(ent->asth->read_pos == ent->asth->write_pos && ent->asth->running){
            pthread_cond_wait(&ent->asth->cond, &ent->asth->mutex);
        }
        
        // Verifica se deve parar
        if (!ent->asth->running && ent->asth->read_pos == ent->asth->write_pos){
            pthread_mutex_unlock(&ent->asth->mutex);
            break;
        }
        
        // Processa TODOS os dados disponíveis
        while(ent->asth->read_pos != ent->asth->write_pos){
            LayerSignal* ls = malloc(sizeof(LayerSignal));
            
            // Lê da posição atual
            ls->outputs = (float*)malloc(3*sizeof(float));
            
            ls->outputs[0] = fabsf(ent->asth->input_buffer[0][ent->asth->read_pos]*100.0f),
            ls->outputs[1] = fabsf(ent->asth->input_buffer[1][ent->asth->read_pos]*100.0f),
            ls->outputs[2] = (float)(ent->asth->label_true*100.0f);
            
            ls->spike_timestamps = (float*)malloc(3*sizeof(float));
            for (int k = 0; k < 3; k++){
                ls->spike_timestamps[k] = dt;
            }
            printf("%f %f\n", ls->outputs[0], ls->outputs[1]);
            
            ls->n_outputs = 3;
            ls->from = 0;
            
            // Avança read_pos
            ent->asth->read_pos = (ent->asth->read_pos + 1) % AUDIO_BUFFER_MAX_SIZE;
            ent->asth->input_buffer_current_size--;
            
            dt += DT;
            pthread_mutex_unlock(&ent->asth->mutex);
            // Libera o mutex ANTES de chamar signal (evita deadlock)
            signal(ent->net, ls, 1);
            pthread_mutex_lock(&ent->asth->mutex);
        }
        pthread_mutex_unlock(&ent->asth->mutex);
    }
    
    return NULL;
}

// Criação CORRIGIDA
AudioStreamThread* create_audio_stream(Network* net, int label_true){
    AudioStreamThread* asth = (AudioStreamThread*)calloc(1, sizeof(AudioStreamThread));
    if (!asth) {
        perror("Failed to allocate AudioStreamThread");
        return NULL;
    }
    
    pthread_cond_init(&asth->cond, NULL);
    pthread_mutex_init(&asth->mutex, NULL);
    asth->running = 1;
    asth->read_pos = 0;
    asth->write_pos = 0;
    asth->input_buffer_current_size = 0;
    asth->label_true = label_true;

    StreamEntry* esth = malloc(sizeof(StreamEntry));
    esth->asth = asth;
    esth->net = net;

    if (pthread_create(&asth->thread, NULL, audio_stream_thread, (void*)esth) != 0) {
        perror("Failed to create audio stream thread");
        pthread_mutex_destroy(&asth->mutex);
        pthread_cond_destroy(&asth->cond);
        free(asth);
        return NULL;
    }
    return asth;
}

// Parada CORRIGIDA
void stop_audio_stream(AudioStreamThread* asth){
    if (!asth) return;
    
    pthread_mutex_lock(&asth->mutex);
    asth->running = 0;
    pthread_cond_signal(&asth->cond);
    pthread_mutex_unlock(&asth->mutex);
    
    pthread_join(asth->thread, NULL);
    
    pthread_mutex_destroy(&asth->mutex);
    pthread_cond_destroy(&asth->cond);
    free(asth);
    printf("Freed asth\n");
}

// Inicialização CORRIGIDA
void init_audio(Network* net, ma_device* device, int label_true){
    // Cria o stream ANTES de iniciar o dispositivo
    asth = create_audio_stream(net, label_true);
    if (!asth) {
        fprintf(stderr, "Failed to create audio stream\n");
        return;
    }
    
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.pDeviceID = NULL;
    config.capture.format = ma_format_f32;
    config.capture.channels = 0;
    config.sampleRate = 0;
    config.dataCallback = data_callback;
    config.periodSizeInFrames = 480;
    config.noPreSilencedOutputBuffer = MA_TRUE;
    config.performanceProfile = ma_performance_profile_low_latency;
    
    if (ma_device_init(NULL, &config, device) != MA_SUCCESS){
        fprintf(stderr, "Could not init audio device\n");
        stop_audio_stream(asth);
        return;
    }

    printf("%s\n", device->capture.name);
    
    if (ma_device_start(device) != MA_SUCCESS){
        fprintf(stderr, "Could not start audio device\n");
        ma_device_uninit(device);
        stop_audio_stream(asth);
        return;
    }
    
    printf("Audio device initialized successfully\n");
    printf("Sample rate: %d, Channels: %d, Format: f32\n", 
           device->sampleRate, device->capture.channels);
}

void end_audio(ma_device* device){
    ma_result r = ma_device_stop(device);
    if (r != MA_SUCCESS){
        perror("Error non end_audio");
        return;
    }

    ma_device_uninit(device);
    printf("Uninit device successful\n");
    stop_audio_stream(asth);
}