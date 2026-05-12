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
            labels[nlayers-1] = label;
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
    }
    
    Network net;
    fread(&net.n_layers, sizeof(int), 1, fptr);

    for (int i = 0; i < net.n_layers; i++){
        printf("%d\n", i);
        net.layers[i] = (Layer*)malloc(sizeof(Layer));

        int label_len;
        char label[200];

        fread(&label_len, sizeof(int), 1, fptr);
        fread(&label, sizeof(char), label_len, fptr);

        printf("AFTER\n");

        net.layers[i]->label = label;
        printf("N_neurons before %d\n", net.layers[i]->n_neurons);
        int result = fread(&net.layers[i]->n_neurons, sizeof(int), 1, fptr);
        if (result == 0){
            perror("Error reading");
        }
        printf("N_neurons %d\n", net.layers[i]->n_neurons);
        net.layers[i]->neurons = (Neuron*)malloc(net.layers[i]->n_neurons*sizeof(Neuron));

        printf("OPA\n");
        fread(&net.layers[i]->idx, sizeof(int), 1, fptr);
        fread(net.layers[i]->neurons, sizeof(Neuron), net.layers[i]->n_neurons, fptr);

        int k;
        int n_input_neurons;
        int count = 0;

        printf("Before crash\n");
        
        for (int l = 0; l < MAX_LAYERS; l++){
            
        }
        fread(&k, sizeof(int), 1, fptr);
        fread(&n_input_neurons, sizeof(int), 1, fptr);
        printf("Opa %d %d\n", k, n_input_neurons);
        net.layers[i]->conns[k] = (float**) malloc(n_input_neurons*sizeof(float*));
        
        printf("After crash 1\n");
        for (int m = 0; m < n_input_neurons; m++){
            
            net.layers[i]->conns[k][m] = (float*) malloc(net.layers[i]->n_neurons * sizeof(float));
            for(int n = 0; n < net.layers[i]->n_neurons; n++){
                fread(&net.layers[i]->conns[k][m][n], sizeof(float), 1, fptr);
            }
            printf("OPAAA %d\n", m);
        }
        
    } 
    printf("AIOSNFA\n");
    return net;
}


void write_neural_bins(const char* filename, Network* net){
    FILE* fptr = fopen(filename, "wb");
    if (!fptr){
        perror("File could not be written");
    }

    fwrite(&net->n_layers, sizeof(int), 1, fptr);
    for (int i = 0; i < net->n_layers; i++){
        int label_len = strlen(net->layers[i]->label);
        fwrite(&label_len, sizeof(int), 1, fptr);
        for (int c = 0; c < label_len; c++){
            fwrite(&net->layers[i]->label[c], sizeof(char), 1, fptr);
        }
        fwrite(&net->layers[i]->n_neurons, sizeof(int), 1, fptr);
        fwrite(&net->layers[i]->idx, sizeof(int), 1, fptr);
        for (int j = 0; j < net->layers[i]->n_neurons; j++){
            fwrite(&net->layers[i]->neurons[j], sizeof(Neuron), 1, fptr);
        }

        int nconns = 0; 
        for (int k = 0; k < MAX_LAYERS; k++){
            if (net->layers[i]->conns[k] && nconns < net->layers[i]->n_conns){
                nconns++;
                fwrite(&k, sizeof(int), 1, fptr);
                fwrite(&net->layers[k]->n_neurons, sizeof(int), 1, fptr);

                for(int l = 0; l < net->layers[k]->n_neurons; l++){
                    for (int m = 0; m < net->layers[i]->n_neurons; m++){
                        fwrite(&net->layers[i]->conns[k][l][m], sizeof(float), 1, fptr);
                
                    }  
                }        
            }
        }
    }
    int result = fflush(fptr);
    if (result != 0){
        perror("Error flushing file: ");
        fclose(fptr);
        return;
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

