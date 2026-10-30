//#include "mtrx_utils.h" Including later
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "snn_lif_stdp.h"
#include "logging/logger.h"
#define SNN_LIF_GENERATION

int write_network_bins(Network* network, const char* filename){
    FILE* file = fopen(filename, "wb");
    if (file == NULL){
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return EXIT_FAILURE;
    }
    // Implementation for writing network bins to file
    fwrite(&network->n_layers, sizeof(int), 1, file);
    for (int i = 0; i < network->n_layers; i++){
        Layer* layer = network->layers[i];
        // Write layer data to file
        fwrite(&layer->idx, sizeof(int), 1, file);
        fwrite(&layer->n_neurons, sizeof(int), 1, file);
        fwrite(layer->label, sizeof(char), strlen(layer->label) + 1, file);
        fwrite(&layer->n_conns, sizeof(int), 1, file);

        for (int j = 0; j < layer->n_neurons; j++){
            // Write neuron data to file
            Neuron* neuron = &layer->neurons[j];
            fwrite(&neuron->current_timestamp, sizeof(float), 1, file);
            fwrite(&neuron->current_u, sizeof(float), 1, file);
            fwrite(&neuron->ref_count, sizeof(float), 1, file);
            fwrite(&neuron->last_spike_timestamp, sizeof(float), 1, file);
        }
    }
    fclose(file);
    return EXIT_SUCCESS;
}

int read_network_bins(Network* network, const char* filename){
    FILE* file = fopen(filename, "rb");
    if (file == NULL){
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return EXIT_FAILURE;
    }
    // Implementation for reading network bins from file
    fread(&network->n_layers, sizeof(int), 1, file);
    for (int i = 0; i < network->n_layers; i++){
        Layer* layer = network->layers[i];
        // Read layer data from file
        fread(&layer->idx, sizeof(int), 1, file);
        fread(&layer->n_neurons, sizeof(int), 1, file);
        layer->label = malloc(100 * sizeof(char)); // Assuming max label length is 100
        fread(layer->label, sizeof(char), 100, file);
        fread(&layer->n_conns, sizeof(int), 1, file);

        for (int j = 0; j < layer->n_neurons; j++){
            // Read neuron data from file
            Neuron* neuron = &layer->neurons[j];
            fread(&neuron->current_timestamp, sizeof(float), 1, file);
            fread(&neuron->current_u, sizeof(float), 1, file);
            fread(&neuron->ref_count, sizeof(float), 1, file);
            fread(&neuron->last_spike_timestamp, sizeof(float), 1, file);
        }
    }
    fclose(file);
    return EXIT_SUCCESS;
}

Network* read_architecture_from_file(const char* filename){
    FILE* file = fopen(filename, "r");
    if (file == NULL){
        fprintf(stderr, "Failed to open architecture file: %s\n", filename);
        return NULL;
    }
    
    int n_layers;
    float conn_prob;

    fscanf(file, "CONN_PROB %f\n", &conn_prob);
    fscanf(file, "N_LAYERS %d\n", &n_layers);

    char* labels[n_layers];
    int n_neurons[n_layers];

    for (int i = 0; i < n_layers; i++){
        int neurons;
        char* label;

        fscanf(file, "%s - %d\n", label, &neurons);
        labels[i] = label;
        n_neurons[i] = neurons;

    }
    Network* tmp = generate_network(n_layers, n_neurons, labels, conn_prob);
    if (tmp == NULL){
        perror("Error in network creation");
        return NULL;
    }
    
    fclose(file);
    return tmp;
}

int main(int argc, char** argv){
    Network* network = calloc(1, sizeof(Network));
    // IMPORTANT! ALWAYS WRITE THE FLAGS and --save AFTER --crfile or --rfile! 
    // see goto operations

    int train = 0;
    char* save_file = NULL;

    for (int i  = 0; i < argc-1; i++){
        if (strcmp(argv[i], "--crfile") == 0){
            const char* filename = argv[i + 1];
            // Reads network from the file
            network = read_architecture_from_file(filename);
            if (network == NULL){
                log_error("Could not read architecture from file");
                return EXIT_FAILURE;
            }

            printf("Network read from %s\n", filename);
            for (int j = i; j < argc-1; j++){
                if (strcmp(argv[j], "--save") == 0){
                    save_file = argv[j+1];
                } else if (strcmp(argv[j], "--train") == 0){
                    train = 1;
                } else if (strcmp(argv[j], "--log") == 0){
                    set_log_file(argv[j+1]);
                }
            } 

            goto operations;

        } else if (strcmp(argv[i], "--rfile") == 0){
            // Reads network from bins
            const char* filename = argv[i + 1];
            int success = read_network_bins(network, filename);
            if (success == EXIT_FAILURE){
                perror("Could not read architecture from file");
                return EXIT_FAILURE;
            }

            printf("Network read from %s\n", filename);
            for (int j = i; j < argc-1; j++){
                if (strcmp(argv[j], "--save") == 0){
                    save_file = argv[j+1];
                } else if (strcmp(argv[j], "--train") == 0){
                    train = 1;
                } else if(strcmp(argv[j], "--log") == 0){
                    set_log_file(argv[j+1]);
                }
            }

            goto operations;
        
        }
    }

    operations:
        // Just writing the architecture again; more changes to come
        if (save_file == NULL){
            log_info("Not saving the network for this run\n");
            return EXIT_SUCCESS;
        }

        if (!train){
            log_info("Not training for this run\n");
        }

        int success = write_network_bins(network, save_file);
        if (success == EXIT_FAILURE){
            log_error("Could not write model bins");
            return EXIT_FAILURE;
        }
        
        free_generated_network(network);
        return EXIT_SUCCESS;

}
