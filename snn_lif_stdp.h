#ifndef SNN_LIF_STDP
    #define SNN_LIF_STDP
    #define MAX_LAYERS 10

    #define TAU_PLUS 1.0f
    #define TAU_MINUS 10.0f
    #define TAU_M 20.0f

    #define U_REST -75.0f
    #define U_TH -50.0f

    #define A_PLUS 0.01f
    #define A_MINUS (A_PLUS)*1.05f

    #define RM 20.0f
    #define REF_PERIOD 2.0f
    #define DT 0.1f

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <string.h>
    #include <time.h>
    #include "logging/logger.h"
    #include "mtrx_utils.h"

    typedef struct {
        float* outputs;
        float* spike_timestamps;
        int n_outputs;
        int from;
    } LayerSignal;

    typedef struct Neuron {
        float current_timestamp;
        float current_u;
        float ref_count;
        float last_spike_timestamp;
        int idx;
    } Neuron;

    typedef struct {
        char* label;
        Neuron* neurons;
        float** conns[MAX_LAYERS];
        int n_conns;
        int n_neurons;
        int idx;
    } Layer;

    typedef struct {
        Layer* layers[MAX_LAYERS];
        int n_layers;
    } Network;

    #if defined(SNN_LIF_GENERATION)
        static char* snn_strdup(const char* source){
            if (source == NULL) return NULL;
            size_t length = strlen(source) + 1;
            char* copy = malloc(length);
            if (copy != NULL) memcpy(copy, source, length);
            return copy;
        }

        static void snn_free_generated_network(Network* net){
            if (net == NULL) return;
            for (int i = 0; i < net->n_layers; i++){
                Layer* layer = net->layers[i];
                if (layer == NULL) continue;
                for (int source = 0; source < MAX_LAYERS; source++){
                    if (layer->conns[source] == NULL) continue;
                    int rows = source < net->n_layers && net->layers[source] != NULL
                        ? net->layers[source]->n_neurons : 0;
                    for (int row = 0; row < rows; row++) free(layer->conns[source][row]);
                    free(layer->conns[source]);
                }
                free(layer->neurons);
                free(layer->label);
                free(layer);
            }
            free(net);
        }

        Network* generate_network(int nlayers, int* nneurons, char** labels, float conn_prob){
            if (nlayers <= 0 || nlayers > MAX_LAYERS || nneurons == NULL || labels == NULL
                || !isfinite(conn_prob) || conn_prob < 0.0f || conn_prob > 1.0f){
                return NULL;
            }
            for (int i = 0; i < nlayers; i++){
                if (nneurons[i] <= 0 || labels[i] == NULL) return NULL;
            }
            Network* net = calloc(1, sizeof(Network));
            if (net == NULL) return NULL;
            net->n_layers = nlayers;
            
            for (int i = 0; i < nlayers; i++){
                net->layers[i] = calloc(1, sizeof(Layer));
                if (net->layers[i] == NULL){
                    snn_free_generated_network(net);
                    return NULL;
                }
                net->layers[i]->n_neurons = nneurons[i];
                net->layers[i]->label = snn_strdup(labels[i]);
                net->layers[i]->idx = i;
                net->layers[i]->neurons = calloc((size_t)nneurons[i], sizeof(Neuron));
                if (net->layers[i]->label == NULL || net->layers[i]->neurons == NULL){
                    snn_free_generated_network(net);
                    return NULL;
                }
                net->layers[i]->n_conns = 0;

                for(int j = 0; j < nneurons[i]; j++){
                    Neuron n = {0.0f, U_REST, 0.0f, 0.0f, 0};
                    net->layers[i]->neurons[j] = n;

                }
            }
            srand(time(NULL));

            for (int i = 0; i < nlayers; i++){
                for(int j = 0; j < nlayers; j++){
                    if (i != j && ((float)rand()/RAND_MAX) < conn_prob){
                        net->layers[j]->conns[i] = calloc((size_t)net->layers[i]->n_neurons, sizeof(float*));
                        if (!net->layers[j]->conns[i]){
                            snn_free_generated_network(net);
                            return NULL;
                        }
                        for (int k = 0; k < net->layers[i]->n_neurons; k++){
                            net->layers[j]->conns[i][k] = malloc(net->layers[j]->n_neurons*sizeof(float));
                            if (!net->layers[j]->conns[i][k]){
                                snn_free_generated_network(net);
                                return NULL;
                            }
                            for(int sinap = 0; sinap < net->layers[j]->n_neurons; sinap++){
                                net->layers[j]->conns[i][k][sinap] = 10.0f*((float)rand()/RAND_MAX);
                            }
                        }
                        net->layers[j]->n_conns++;
                    }
                }
            }

            printf("Finished\n");
            return net;
        }

    #endif

    #if defined(SNN_LIF_PROC)
        LayerSignal LIF_Signal(LayerSignal input, int lif_only, Layer* layer, int nneurons_out, float** w[]){
            LayerSignal output = {0};
            if (input.outputs == NULL || layer == NULL || nneurons_out <= 0 || w == NULL
                || input.from < 0 || input.from >= MAX_LAYERS || w[input.from] == NULL){
                return output;
            }
            output.n_outputs = nneurons_out;
            output.outputs = calloc(nneurons_out, sizeof(float));
            output.spike_timestamps = calloc(nneurons_out, sizeof(float));
            if (output.outputs == NULL || output.spike_timestamps == NULL){
                free(output.outputs);
                free(output.spike_timestamps);
                return (LayerSignal){0};
            }
            output.from = layer->idx;

            for (int j = 0; j < nneurons_out; j++){
                Neuron* n = &layer->neurons[j];
                float current = 0.0f;
                for (int i = 0; i < input.n_outputs; i++){
                    current += input.outputs[i] * w[input.from][i][j];
                }
                float du = (-(n->current_u - U_REST) + RM*current) * (DT/TAU_M);
                n->current_u += du;
                n->current_timestamp += DT;

                if(n->current_u >= U_TH){
                    output.outputs[j] = lif_only > 0 ? 70.0f : 70.0f;
                    output.spike_timestamps[j] = n->current_timestamp - DT * 0.5f;

                    Data d = {n->idx, layer->idx, n->current_u, n->current_timestamp};
                    n->current_u = U_REST;
                    n->last_spike_timestamp = n->current_timestamp;
                    logger_log(EVENT_TYPE_SPIKE, d);
                }
            }
            return output;
        }
        void STDP(LayerSignal ls, Layer* layer){
            int from = ls.from;
            if (layer == NULL || ls.outputs == NULL || ls.spike_timestamps == NULL
                || from < 0 || from >= MAX_LAYERS || layer->conns[from] == NULL) return;
            for (int i = 0; i < ls.n_outputs; i++){
                if (ls.outputs == NULL || ls.outputs[i] <= 0.0f) continue;
                float spike_tms = ls.spike_timestamps[i];
                for (int j = 0; j < layer->n_neurons; j++){
                    if (layer->neurons[j].last_spike_timestamp <= 0.0f) continue;
                    float dt = layer->neurons[j].last_spike_timestamp - spike_tms;
                    float dw = 0.0f;
                    if (dt > 0){
                        dw = A_PLUS * exp(-dt/TAU_PLUS);
                    } else if (dt < 0) {
                        dw = -A_MINUS * exp(dt/TAU_MINUS);
                    }
                    float* w = &layer->conns[from][i][j];
                    float updated_weight = *w + dw;
                    updated_weight = updated_weight > 3.0f ? 3.0f : updated_weight;
                    updated_weight = updated_weight < -3.0f ? -3.0f : updated_weight;
                    if (updated_weight == *w) continue;
                    *w = updated_weight;

                    Data data = {i, layer->idx, updated_weight, layer->neurons[j].current_timestamp};
                    logger_log(EVENT_TYPE_WEIGTH_UPDATE, data);

                }

            }

        }
    #endif
#endif