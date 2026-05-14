#define MINIAUDIO_IMPLEMENTATION
#define MA_DEBUG_OUTPUT
#include "miniaudio.h"
#include "astream.h"
#include "../snn_lif_stdp.h"
#include "../threads/layer_thread.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount){
    // Verificação de segurança
    if (pInput == NULL) return;
    
    const float* pInputFloat = (const float*) pInput;
    int channels = pDevice->capture.channels;
    static float dt = 0.0f;  // static para manter entre callbacks
    
    // Itera sobre FRAMES, não sobre samples individuais
    for(ma_uint32 frame = 0; frame < frameCount; frame++){
        // Aloca UMA VEZ por frame (idealmente deveria ser pré-alocado)
        LayerSignal ls;
        ls.outputs = (float*)malloc(2 * sizeof(float));
        ls.spike_timestamps = (float*)malloc(2 * sizeof(float));
        
        if (!ls.outputs || !ls.spike_timestamps) {
            // Falha na alocação - libera o que foi alocado
            free(ls.outputs);
            free(ls.spike_timestamps);
            continue;  // Ou retorna para não travar o áudio
        }
        
        ls.n_outputs = 2;
        ls.from = MAX_LAYERS;

        // CORRIGIDO: Acessa samples corretamente
        // frame * channels = índice do primeiro canal do frame
        int base_idx = frame * channels;
        ls.outputs[0] = pInputFloat[base_idx] * 50.0f;      // Canal esquerdo
        ls.outputs[1] = pInputFloat[base_idx + 1] * 50.0f;  // Canal direito

        printf("%f %f\n", ls.outputs[0], ls.outputs[1]);
        ls.spike_timestamps[0] = dt;
        ls.spike_timestamps[1] = dt;
        
        dt += DT;
        
        signal(ls);
        
        // Libera após o sinal ser processado
        // NOTA: Se signal() copia os dados, pode liberar aqui
        // Se signal() apenas armazena o ponteiro, NÃO libere aqui!
        free(ls.outputs);
        free(ls.spike_timestamps);
    }
}

void init_audio(ma_device* device){
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    
    // Configurações de captura
    config.capture.pDeviceID = NULL;  // Dispositivo padrão
    config.capture.format = ma_format_f32;  // 16-bit signed integer
    config.capture.channels = 0;  // Estéreo
    config.sampleRate = 0;  // 44100 Hz
    config.dataCallback = data_callback;
    
    // Configurações do período (opcional, mas recomendado)
    config.periodSizeInFrames = 512;  // Ajuste conforme necessidade
    
    if (ma_device_init(NULL, &config, device) != MA_SUCCESS){
        fprintf(stderr, "Could not init audio device\n");
        return;
    }

    printf("%s\n", device->capture.name);
    if (ma_device_start(device) != MA_SUCCESS){
        fprintf(stderr, "Could not start audio device\n");
        ma_device_uninit(device);
        return;
    }
    
    printf("Audio device initialized successfully\n");
    printf("Sample rate: %d, Channels: %d, Format: s16\n", 
           device->sampleRate, device->capture.channels);
}

void end_audio(ma_device* device){
    ma_device_stop(device);
}
