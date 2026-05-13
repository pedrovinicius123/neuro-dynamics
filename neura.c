#define SNN_LIF_GENERATION
#include "snn_lif_stdp.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

Network* read_neural_architecture(const char* filename){
    FILE* fptr = fopen(filename, "r");
    if (!fptr){
        perror("Error opening file: (line 6 - neura.c)");
        return NULL;
    }

    char line[100];
    int* n_neurons = malloc(sizeof(int));
    char** labels = malloc(sizeof(char*));
    int nlayers = 0;
    float conn_prob;
    
    while (fgets(line, sizeof(line), fptr)){
        char label[100];
        int nneuron;
        if(line[0] == '#'){
            sscanf(line, "# %f", &conn_prob);
        } else {
            sscanf(line, "(%[^)]) - %d", label,  &nneuron);
            labels[nlayers-1] = strdup(label);
            n_neurons[nlayers-1] = nneuron;
            
            labels = (char**)realloc(labels, (nlayers+1)*sizeof(char*));
            n_neurons = (int*)realloc(n_neurons, (nlayers+1)*sizeof(int));
            //sleep(1);
        }
        nlayers++;
        
    }   
    return generate_network(nlayers-1, n_neurons, labels, conn_prob);
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
            
            // Aloca todos os dados contiguamente
            float* data = (float*)malloc(nneurs * net.layers[i]->n_neurons * sizeof(float));
            
            // Lê tudo de uma vez
            fread(data, sizeof(float), nneurs * net.layers[i]->n_neurons, fptr);
            
            // Configura os ponteiros para cada linha
            for (int row = 0; row < nneurs; row++) {
                net.layers[i]->conns[k][row] = &data[row * net.layers[i]->n_neurons];
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

int main(int argc, char** argv){
    char* create_filename = NULL;
    char* save_filename = NULL;
    char* read_filename = NULL;
    int debug = 0;

    for (int i = 0; i < argc; i++){
        if(strcmp(argv[i], "--crfile") == 0 && i != argc-1){
            create_filename = argv[i+1];
        } else if (strcmp(argv[i], "--debug") == 0){
            debug = 1;
        } else if(strcmp(argv[i], "--sfile") == 0 && i != argc-1){
            save_filename = argv[i+1];
        } else if(strcmp(argv[i], "--rfile") == 0 && i != argc-1){
            read_filename = argv[i+1];
        }
    }

    if (debug){
        printf("DEBUG: Reading file %s\n", create_filename);
    }

    Network* l = malloc(sizeof(Network));
    if (create_filename == NULL ^ read_filename == NULL){
        if (create_filename){
            l = read_neural_architecture(create_filename);
            if (!l){
                return 1;
            }
        } else {
            Network net = read_neural_bins(read_filename);
            l = &net;
            if (!l){
                return 1;
            }
        }
        if (debug){
            printf("DEBUG: Layer model read\n");
            for (int layer = 0; layer < l->n_layers; layer++){
                for (int i = 0; i < MAX_LAYERS; i++){
                    if (l->layers[layer]->conns[i] != NULL){
                        printf("LAYER %d to LAYER %d - OK\n", layer, i);
                    }
                }            
            }
        }

        if (save_filename){
            write_neural_bins(save_filename, l);
        }
    } else {
        printf("Error, --rfile xor --sfile => 0\n");
        return 1;
    }
    return 0;
}

