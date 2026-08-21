#define SNN_LIF_GENERATION
#include "snn_lif_stdp.h"
#include "threads/layer_thread.h"
#include "audio/astream.h"
#include "audio/miniaudio.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void free_network_contents(Network* net){
    if (net == NULL) return;
    for (int i = 0; i < net->n_layers; i++){
        Layer* layer = net->layers[i];
        if (layer == NULL) continue;
        for (int from = 0; from < MAX_LAYERS; from++){
            if (layer->conns[from] == NULL) continue;
            for (int row = 0; row < net->layers[from]->n_neurons; row++){
                free(layer->conns[from][row]);
            }
            free(layer->conns[from]);
        }
        }
        for (int i = 0; i < net->n_layers; i++){
            Layer* layer = net->layers[i];
            if (layer == NULL) continue;
        free(layer->neurons);
        free(layer->label);
        free(layer);
    }
}

static void free_network(Network* net){
    free_network_contents(net);
    free(net);
}

Network* read_neural_architecture(const char* filename){
    FILE* fptr = fopen(filename, "r");
    if (!fptr){
        perror("Error opening file: (line 6 - neura.c)");
        return NULL;
    }

    char line[100];
    int* n_neurons = malloc(MAX_LAYERS * sizeof(int));
    char** labels = malloc(MAX_LAYERS * sizeof(char*));
    int nlayers = 0;
    float conn_prob = 0.0f;
    if (n_neurons == NULL || labels == NULL){
        free(n_neurons);
        free(labels);
        fclose(fptr);
        return NULL;
    }
    
    while (fgets(line, sizeof(line), fptr)){
        char label[100];
        int nneuron;
        if (line[0] == '#') {
            if (sscanf(line, "# %f", &conn_prob) != 1) goto invalid;
        } else if (sscanf(line, "(%[^)]) - %d", label, &nneuron) == 2) {
            if (nlayers >= MAX_LAYERS || nneuron <= 0) goto invalid;
            char* parsed_label = snn_strdup(label);
            if (parsed_label == NULL) goto invalid;
            labels[nlayers] = parsed_label;
            n_neurons[nlayers] = nneuron;
            nlayers++;
        } else if (line[0] != '\n' && line[0] != '\r' && line[0] != '\0') {
            goto invalid;
        }
    }   
    fclose(fptr);
    if (nlayers == 0) goto invalid_after_close;
    Network* network = generate_network(nlayers, n_neurons, labels, conn_prob);
    for (int i = 0; i < nlayers; i++) free(labels[i]);
    free(labels);
    free(n_neurons);
    return network;

invalid:
    fclose(fptr);
invalid_after_close:
    for (int i = 0; i < nlayers; i++) free(labels[i]);
    free(labels);
    free(n_neurons);
    fprintf(stderr, "Invalid neural architecture: %s\n", filename);
    return NULL;
}

Network read_neural_bins(const char* filename){
    FILE* fptr = fopen(filename, "rb");
    if (!fptr){
        perror("File could not be read");
        exit(1);
    }
    
    Network net;
    memset(&net, 0, sizeof(Network));  // Inicializa tudo com zero
    
    fread(&net.n_layers, sizeof(int), 1, fptr);
    printf("n_layers: %d\n", net.n_layers);
    
    for (int i = 0; i < net.n_layers; i++){
        int label_len;
        
        net.layers[i] = (Layer*)calloc(1, sizeof(Layer));  // calloc inicializa tudo com NULL/0
        //printf("Layer %d\n", i);

        fread(&label_len, sizeof(int), 1, fptr);

        char* label = (char*)malloc(label_len + 1);  // +1 para null terminator
        fread(label, sizeof(char), label_len, fptr);
        label[label_len] = '\0';  // Adiciona terminador
        //printf("Label: %s\n", label);
        
        net.layers[i]->label = label;  // Já foi alocado, não precisa de strdup
        
        fread(&net.layers[i]->n_neurons, sizeof(int), 1, fptr);
        //printf("n_neurons: %d\n", net.layers[i]->n_neurons);

        fread(&net.layers[i]->idx, sizeof(int), 1, fptr);
        //printf("idx: %d\n", net.layers[i]->idx);
    
        // ALOCA e lê os neurônios
        net.layers[i]->neurons = (Neuron*)malloc(net.layers[i]->n_neurons * sizeof(Neuron));
        fread(net.layers[i]->neurons, sizeof(Neuron), net.layers[i]->n_neurons, fptr);
        
        for (int n = 0; n < net.layers[i]->n_neurons; n++){
            net.layers[i]->neurons[n].idx = n;
        }

        int nconns;
        fread(&nconns, sizeof(int), 1, fptr);
        ////printf("nconns: %d\n", nconns);

        // Inicializa conns com NULL
        memset(net.layers[i]->conns, 0, sizeof(float**) * MAX_LAYERS);

        for (int nc = 0; nc < nconns; nc++){
            int nneurs;
            int k;
            fread(&k, sizeof(int), 1, fptr);
            fread(&nneurs, sizeof(int), 1, fptr);
            
            //printf("Connection %d: k=%d, nneurs=%d\n", nc, k, nneurs);
            
            // Aloca array de ponteiros para linhas
            net.layers[i]->conns[k] = (float**)malloc(nneurs * sizeof(float*));
            
            for (int row = 0; row < nneurs; row++) {
                net.layers[i]->conns[k][row] = malloc(net.layers[i]->n_neurons * sizeof(float));
                if (net.layers[i]->conns[k][row] == NULL ||
                    fread(net.layers[i]->conns[k][row], sizeof(float), net.layers[i]->n_neurons, fptr) != (size_t)net.layers[i]->n_neurons) {
                    fclose(fptr);
                    free_network_contents(&net);
                    return (Network){0};
                }
            }
        }
    }

    fclose(fptr);
    return net;
}

void write_neural_bins(const char* filename, Network* net){
    FILE* fptr = fopen(filename, "wb");
    if (!fptr){
        perror("File could not be written");
        return;
    }

    printf("NET LAYERS %d\n", net->n_layers);
    
    fwrite(&net->n_layers, sizeof(int), 1, fptr);
    
    for (int i = 0; i < net->n_layers; i++){
        int label_len = strlen(net->layers[i]->label);
        //printf("%d Label %s\n", label_len, net->layers[i]->label);

        fwrite(&label_len, sizeof(int), 1, fptr);
        fwrite(net->layers[i]->label, sizeof(char), label_len, fptr);
        fwrite(&net->layers[i]->n_neurons, sizeof(int), 1, fptr);
        fwrite(&net->layers[i]->idx, sizeof(int), 1, fptr);
        fwrite(net->layers[i]->neurons, sizeof(Neuron), net->layers[i]->n_neurons, fptr);

        // Conta conexões existentes primeiro
        int nconns = 0;
        for (int k = 0; k < MAX_LAYERS; k++){
            if (net->layers[i]->conns[k]) nconns++;
        }
        net->layers[i]->n_conns = nconns;
        
        fwrite(&nconns, sizeof(int), 1, fptr);
        //printf("nconns: %d\n", nconns);

        for (int k = 0; k < MAX_LAYERS; k++){
            if (net->layers[i]->conns[k]){
                fwrite(&k, sizeof(int), 1, fptr);
                // CORRIGIDO: número de linhas da matriz (neurônios da camada k)
                fwrite(&net->layers[k]->n_neurons, sizeof(int), 1, fptr);
                
                // Escreve a matriz: linhas = net->layers[k]->n_neurons, colunas = net->layers[i]->n_neurons
                for(int l = 0; l < net->layers[k]->n_neurons; l++){
                    fwrite(net->layers[i]->conns[k][l], sizeof(float), net->layers[i]->n_neurons, fptr);
                }
            }
        }
        fflush(fptr);
    }
    
    fclose(fptr);
}

#ifndef NEURA_NO_MAIN
int main(int argc, char** argv){
    char* create_filename = NULL;
    char* save_filename = NULL;
    char* read_filename = NULL;
    int label_true = 0;
    int debug = 0;

    int record_audio = 0;
    for (int i = 1; i < argc; i++){
        if(strcmp(argv[i], "--crfile") == 0 && i != argc-1){
            create_filename = argv[i+1];
        } else if (strcmp(argv[i], "--debug") == 0){
            debug = 1;
        } else if(strcmp(argv[i], "--sfile") == 0 && i != argc-1){
            save_filename = argv[i+1];
        } else if(strcmp(argv[i], "--rfile") == 0 && i != argc-1){
            read_filename = argv[i+1];
        } else if(strcmp(argv[i], "--train") == 0 && i != argc-1){
            label_true = 1;
        } else if (strcmp(argv[i], "--audio") == 0){
            record_audio = 1;
        }
    }

    if (debug && create_filename){
        printf("DEBUG: Reading file %s\n", create_filename);
    } else if(debug){
        printf("DEBUG: Reading file %s\n", read_filename);
    }

    Network* l = NULL;
    if ((create_filename != NULL) ^ (read_filename != NULL)){
        if (create_filename){
            l = read_neural_architecture(create_filename);
            if (!l){
                return 1;
            }
        } else {
            l = (Network*)malloc(sizeof(Network));
            (*l) = read_neural_bins(read_filename);
            if (!l){
                return 1;
            }
        }
        if (debug && l != NULL){
            printf("DEBUG: Layer model read\n");

            for (int layer = 0; layer < l->n_layers; layer++){
                for (int i = 0; i < MAX_LAYERS; i++){
                    if (l->layers[layer]->conns[i] != NULL){
                        printf("SAMPLE: %f\n", l->layers[layer]->conns[i][1][1]);
                        printf("LAYER %d to LAYER %d - OK\n", layer, i);
                    }
                }            
            }
            sleep(2);
        }

        if (record_audio) {
            ma_device device;
            init_logger("logs/log.1.neur");
            NetworkThreads* net_ts = init_layer_threads(l);
            init_audio(l, &device, label_true);
            printf("Recording audio on machine...\n");
            sleep(2);
            end_audio(&device);
            end_layer_threads(net_ts);
            logger_end();
        } else {
            printf("Model loaded: %d layers\n", l->n_layers);
        }
        if (save_filename){
            write_neural_bins(save_filename, l);
        }
        free_network(l);
    } else {
        printf("Error, --rfile xor --sfile => 0\n");
        return 1;
    }
    return 0;
}
#endif

