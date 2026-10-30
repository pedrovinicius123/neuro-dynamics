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

static AudioStreamThread* active_audio_stream;
static int audio_device_initialized;
static int audio_device_started;

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount){
    (void)pOutput;
    if (pInput == NULL || active_audio_stream == NULL) return;

    const float* input = (const float*)pInput;
    int channels = pDevice->capture.channels;
    if (channels <= 0) return;

    pthread_mutex_lock(&active_audio_stream->mutex);
    if (active_audio_stream->running){
        for (ma_uint32 frame = 0; frame < frameCount; frame++){
            int base_index = (int)frame * channels;
            if (active_audio_stream->input_buffer_current_size == AUDIO_BUFFER_MAX_SIZE){
                active_audio_stream->read_pos = (active_audio_stream->read_pos + 1) % AUDIO_BUFFER_MAX_SIZE;
                active_audio_stream->input_buffer_current_size--;
            }
            active_audio_stream->input_buffer[0][active_audio_stream->write_pos] = input[base_index];
            active_audio_stream->input_buffer[1][active_audio_stream->write_pos] = channels > 1
                ? input[base_index + 1] : input[base_index];
            active_audio_stream->write_pos = (active_audio_stream->write_pos + 1) % AUDIO_BUFFER_MAX_SIZE;
            active_audio_stream->input_buffer_current_size++;
        }
        pthread_cond_signal(&active_audio_stream->cond);
    }
    pthread_mutex_unlock(&active_audio_stream->mutex);
}

void* audio_stream_thread(void* args){
    StreamEntry* ent = (StreamEntry*)args;
    float dt = 0.0f;
    for (;;){
        pthread_mutex_lock(&ent->asth->mutex);
        while (ent->asth->input_buffer_current_size == 0 && ent->asth->running){
            pthread_cond_wait(&ent->asth->cond, &ent->asth->mutex);
        }
        if (!ent->asth->running && ent->asth->input_buffer_current_size == 0){
            pthread_mutex_unlock(&ent->asth->mutex);
            break;
        }

        int position = ent->asth->read_pos;
        float left = ent->asth->input_buffer[0][position];
        float right = ent->asth->input_buffer[1][position];
        int label_true = ent->asth->label_true;
        ent->asth->read_pos = (position + 1) % AUDIO_BUFFER_MAX_SIZE;
        ent->asth->input_buffer_current_size--;
        pthread_mutex_unlock(&ent->asth->mutex);

        LayerSignal* ls = calloc(1, sizeof(LayerSignal));
        if (ls != NULL){
            ls->outputs = calloc(3, sizeof(float));
            ls->spike_timestamps = calloc(3, sizeof(float));
            if (ls->outputs == NULL || ls->spike_timestamps == NULL){
                free(ls->outputs);
                free(ls->spike_timestamps);
                free(ls);
            } else {
                ls->outputs[0] = fabsf(left * 100.0f);
                ls->outputs[1] = fabsf(right * 100.0f);
                ls->outputs[2] = (float)(label_true * 100.0f);
                for (int k = 0; k < 3; k++) ls->spike_timestamps[k] = dt;
                ls->n_outputs = 3;
                ls->from = 0;
                signal(ent->net, ls, 1);
                free(ls->outputs);
                free(ls->spike_timestamps);
                free(ls);
            }
        }
        dt += DT;
    }
    free(ent);
    return NULL;
}

AudioStreamThread* create_audio_stream(Network* net, int label_true){
    if (net == NULL) return NULL;
    AudioStreamThread* asth = calloc(1, sizeof(AudioStreamThread));
    if (asth == NULL) return NULL;
    if (pthread_mutex_init(&asth->mutex, NULL) != 0){
        free(asth);
        return NULL;
    }
    if (pthread_cond_init(&asth->cond, NULL) != 0){
        pthread_mutex_destroy(&asth->mutex);
        free(asth);
        return NULL;
    }
    asth->running = 1;
    asth->label_true = label_true;

    StreamEntry* esth = malloc(sizeof(StreamEntry));
    if (esth == NULL){
        pthread_mutex_destroy(&asth->mutex);
        pthread_cond_destroy(&asth->cond);
        free(asth);
        return NULL;
    }
    esth->asth = asth;
    esth->net = net;

    if (pthread_create(&asth->thread, NULL, audio_stream_thread, (void*)esth) != 0) {
        free(esth);
        pthread_mutex_destroy(&asth->mutex);
        pthread_cond_destroy(&asth->cond);
        free(asth);
        return NULL;
    }
    return asth;
}

void stop_audio_stream(AudioStreamThread* asth){
    if (!asth) return;
    
    pthread_mutex_lock(&asth->mutex);
    asth->running = 0;
    pthread_cond_signal(&asth->cond);
    pthread_mutex_unlock(&asth->mutex);
    pthread_join(asth->thread, NULL);
    pthread_mutex_destroy(&asth->mutex);
    pthread_cond_destroy(&asth->cond);
    if (active_audio_stream == asth) active_audio_stream = NULL;
    free(asth);
    printf("Freed asth\n");
}

void init_audio(Network* net, ma_device* device, int label_true){
    if (device == NULL) return;
    active_audio_stream = create_audio_stream(net, label_true);
    if (!active_audio_stream) {
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
        stop_audio_stream(active_audio_stream);
        return;
    }
    audio_device_initialized = 1;

    printf("%s\n", device->capture.name);
    
    if (ma_device_start(device) != MA_SUCCESS){
        fprintf(stderr, "Could not start audio device\n");
        ma_device_uninit(device);
        audio_device_initialized = 0;
        stop_audio_stream(active_audio_stream);
        return;
    }
    audio_device_started = 1;
    
    printf("Audio device initialized successfully\n");
    printf("Sample rate: %d, Channels: %d, Format: f32\n", 
           device->sampleRate, device->capture.channels);
}

void end_audio(ma_device* device){
    if (audio_device_started && device != NULL){
        ma_result result = ma_device_stop(device);
        if (result != MA_SUCCESS) fprintf(stderr, "Could not stop audio device\n");
        audio_device_started = 0;
    }
    if (audio_device_initialized && device != NULL){
        ma_device_uninit(device);
        audio_device_initialized = 0;
    }
    if (active_audio_stream != NULL) stop_audio_stream(active_audio_stream);
}