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
        Network* generate_network(int nlayers, int* nneurons, char** labels, float conn_prob){
            Network* net = malloc(sizeof(Network));
            net->n_layers = nlayers;
            
            for (int i = 0; i < nlayers; i++){
                net->layers[i] = malloc(sizeof(Layer));
                net->layers[i]->n_neurons = nneurons[i];
                net->layers[i]->label = strdup(labels[i]);
                net->layers[i]->idx = i;
                net->layers[i]->neurons = malloc(nneurons[i]*sizeof(Neuron));
                net->layers[i]->n_conns = 0;

                for(int j = 0; j < nneurons[i]; j++){
                    Neuron n = {0.0f, U_REST, 0.0f, 0.0f, 0};
                    net->layers[i]->neurons[j] = n;

                }
                for (int j = 0; j < nlayers; j++){
                    net->layers[i]->conns[j] = NULL;
            
                }
            }
            srand(time(NULL));

            for (int i = 0; i < nlayers; i++){
                for(int j = 0; j < nlayers; j++){
                    if (i != j && ((float)rand()/RAND_MAX) < conn_prob){
                        if (net->layers[i]->conns[j] != NULL){
                            int rows = net->layers[i]->n_neurons;
                            int cols = net->layers[j]->n_neurons;
                            
                            //printf("TRANSPOSED\n");

                            net->layers[j]->n_conns++;
                            net->layers[j]->conns[i] = transpose(net->layers[i]->conns[j], rows, cols);


                            if (!net->layers[j]->conns[i]) {
                                perror("Error while generating SNN (transpose failed)");
                                return NULL;
                            }
                            continue;
                        }
                        net->layers[j]->n_conns++;
                        net->layers[j]->conns[i] = malloc(net->layers[i]->n_neurons*sizeof(float*));
                        if (!net->layers[j]->conns[i]){
                            perror("Error while generating SNN (line 65 - snn_lif_stdp.h)");
                            return NULL;
                        }
                        for (int k = 0; k < net->layers[i]->n_neurons; k++){
                            net->layers[j]->conns[i][k] = malloc(net->layers[j]->n_neurons*sizeof(float));
                            if (!net->layers[j]->conns[i][k]){
                                perror("Error while generating SNN (line 73 - snn_lif_stdp.h)");
                                return NULL;
                            }
                            for(int sinap = 0; sinap < net->layers[j]->n_neurons; sinap++){
                                net->layers[j]->conns[i][k][sinap] = 10.0f*((float)rand()/RAND_MAX);
                            }
                        }
                    } else net->layers[j]->conns[i] = NULL;
                }
            }

            for (int i = 0; i < nlayers; i++){
                for (int j = net->layers[i]->n_conns; j < MAX_LAYERS; j++){
                    net->layers[i]->conns[j] = NULL;
                }                
            }
            printf("Finished\n");
            return net;
        }

    #endif

    #if defined(SNN_LIF_PROC)
        LayerSignal LIF_Signal(LayerSignal input, int lif_only, Layer* layer, int nneurons_out, float** w[]){
            LayerSignal output;
            output.n_outputs = nneurons_out;
            output.outputs = calloc(nneurons_out, sizeof(float));
            output.spike_timestamps = calloc(nneurons_out, sizeof(float));
            output.from = layer->idx;

            for (int i = 0; i < input.n_outputs; i++){
                Neuron* n = &layer->neurons[i];
                float du = (-(n->current_u - U_REST) + RM*input.outputs[i]) * (DT/TAU_M);
                n->current_u += du;
                n->current_timestamp += DT;

                for (int j = 0; j < nneurons_out; j++){
                    if(n->current_u >= U_TH){
                        printf("SPIKE!!\n");
                        output.outputs[i] += lif_only > 0 ? 70.0f : 70.0f * w[input.from][i][j];
                        output.spike_timestamps[i] = n->current_timestamp;

                        Data d = {
                            n->idx,
                            layer->idx,
                            n->current_u,
                            n->last_spike_timestamp
                        };

                        n->current_u = U_REST;
                        n->last_spike_timestamp = n->current_timestamp;
                        
                        logger_log(EVENT_TYPE_SPIKE, d);
                    } 
                }
            }
            return output;
        }
        void STDP(LayerSignal ls, Layer* layer){
            int from = ls.from;
            for (int i = 0; i < ls.n_outputs; i++){
                float spike_tms = ls.spike_timestamps[i];
                for (int j = 0; j < layer->n_neurons; j++){
                    float dt = layer->neurons[j].last_spike_timestamp - spike_tms;
                    float dw = 0.0f;
                    if (dt > 0){
                        dw = A_PLUS * exp(-dt/TAU_PLUS);
                    } else if (dt < 0) {
                        dw = -A_MINUS * exp(dt/TAU_MINUS);
                    }
                    layer->conns[from][i][j] += dw;

                    float* w = &layer->conns[from][i][j];
                    (*w) = *w > 3.0f ? 3.0f :*w;
                    (*w) = *w < -3.0f ?- 3.0f : *w;

                    Data data = {layer->idx, 0, 0.0f, layer->neurons[0].current_timestamp};
                    logger_log(EVENT_TYPE_WEIGTH_UPDATE, data);

                }

            }

        }
    #endif
#endif