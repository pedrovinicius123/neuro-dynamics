#define SNN_LIF_GENERATION
#include "snn_lif_stdp.h"
#include "threads/layer_thread.h"
#include "audio/astream.h"
#include "audio/miniaudio.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

static void free_network_contents(Network* net){
    if (net == NULL) return;
    int n_layers = net->n_layers;
    if (n_layers < 0 || n_layers > MAX_LAYERS) n_layers = MAX_LAYERS;
    for (int i = 0; i < n_layers; i++){
        Layer* layer = net->layers[i];
        if (layer == NULL) continue;
        for (int from = 0; from < MAX_LAYERS; from++){
            if (layer->conns[from] == NULL) continue;
            int rows = from < n_layers && net->layers[from] != NULL
                ? net->layers[from]->n_neurons : 0;
            for (int row = 0; row < rows; row++){
                free(layer->conns[from][row]);
            }
            free(layer->conns[from]);
            layer->conns[from] = NULL;
        }
    }
    for (int i = 0; i < n_layers; i++){
        Layer* layer = net->layers[i];
        if (layer == NULL) continue;
        free(layer->neurons);
        free(layer->label);
        free(layer);
        net->layers[i] = NULL;
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

static int read_model_bytes(const unsigned char* data, size_t size,
                            size_t* offset, void* output, size_t count){
    if (*offset > size || count > size - *offset) return 0;
    memcpy(output, data + *offset, count);
    *offset += count;
    return 1;
}

Network read_neural_bins(const char* filename){
    FILE* fptr = fopen(filename, "rb");
    if (!fptr){
        perror("File could not be read");
        return (Network){0};
    }

    if (fseek(fptr, 0, SEEK_END) != 0){
        fclose(fptr);
        return (Network){0};
    }
    long file_length = ftell(fptr);
    if (file_length < (long)sizeof(int) || fseek(fptr, 0, SEEK_SET) != 0){
        fclose(fptr);
        return (Network){0};
    }
    size_t file_size = (size_t)file_length;
    unsigned char* data = malloc(file_size);
    if (data == NULL || fread(data, 1, file_size, fptr) != file_size){
        free(data);
        fclose(fptr);
        return (Network){0};
    }
    fclose(fptr);

    size_t offset = 0;
    int n_layers;
    int neuron_counts[MAX_LAYERS] = {0};
    int connection_rows[MAX_LAYERS][MAX_LAYERS];
    memset(connection_rows, 0xff, sizeof(connection_rows));
    int valid = read_model_bytes(data, file_size, &offset, &n_layers, sizeof(int))
        && n_layers > 0 && n_layers <= MAX_LAYERS;

    for (int i = 0; valid && i < n_layers; i++){
        int label_len, neuron_count, layer_idx, nconns;
        valid = read_model_bytes(data, file_size, &offset, &label_len, sizeof(int))
            && label_len > 0 && label_len <= 4096
            && (size_t)label_len <= file_size - offset;
        if (!valid) break;
        offset += (size_t)label_len;
        valid = read_model_bytes(data, file_size, &offset, &neuron_count, sizeof(int))
            && read_model_bytes(data, file_size, &offset, &layer_idx, sizeof(int))
            && neuron_count > 0 && neuron_count <= 1000000 && layer_idx == i;
        if (!valid) break;
        neuron_counts[i] = neuron_count;
        size_t neuron_bytes = (size_t)neuron_count * sizeof(Neuron);
        valid = neuron_bytes / sizeof(Neuron) == (size_t)neuron_count
            && offset <= file_size && neuron_bytes <= file_size - offset;
        if (!valid) break;
        offset += neuron_bytes;
        valid = read_model_bytes(data, file_size, &offset, &nconns, sizeof(int))
            && nconns >= 0 && nconns < n_layers;
        for (int nc = 0; valid && nc < nconns; nc++){
            int source, rows;
            valid = read_model_bytes(data, file_size, &offset, &source, sizeof(int))
                && read_model_bytes(data, file_size, &offset, &rows, sizeof(int))
                && source >= 0 && source < n_layers && source != i
                && rows > 0 && rows <= 1000000
                && connection_rows[i][source] == -1;
            if (!valid) break;
            connection_rows[i][source] = rows;
            size_t values = (size_t)rows * (size_t)neuron_count;
            if (values / (size_t)neuron_count != (size_t)rows
                || values > (file_size - offset) / sizeof(float)){
                valid = 0;
                break;
            }
            offset += values * sizeof(float);
        }
    }
    if (valid && offset != file_size) valid = 0;
    for (int target = 0; valid && target < n_layers; target++){
        for (int source = 0; source < n_layers; source++){
            if (connection_rows[target][source] != -1
                && connection_rows[target][source] != neuron_counts[source]){
                valid = 0;
                break;
            }
        }
    }
    if (!valid){
        free(data);
        fprintf(stderr, "Invalid or truncated model: %s\n", filename);
        return (Network){0};
    }

    Network net;
    memset(&net, 0, sizeof(net));
    net.n_layers = n_layers;
    for (int i = 0; i < n_layers; i++){
        net.layers[i] = calloc(1, sizeof(Layer));
        if (net.layers[i] == NULL){
            free_network_contents(&net);
            free(data);
            return (Network){0};
        }
        net.layers[i]->n_neurons = neuron_counts[i];
        net.layers[i]->idx = i;
    }

    offset = sizeof(int);
    for (int i = 0; i < n_layers; i++){
        int label_len, nconns;
        read_model_bytes(data, file_size, &offset, &label_len, sizeof(int));
        net.layers[i]->label = malloc((size_t)label_len + 1);
        if (net.layers[i]->label == NULL){
            free_network_contents(&net);
            free(data);
            return (Network){0};
        }
        read_model_bytes(data, file_size, &offset, net.layers[i]->label, (size_t)label_len);
        net.layers[i]->label[label_len] = '\0';
        offset += sizeof(int) * 2;
        net.layers[i]->neurons = malloc((size_t)neuron_counts[i] * sizeof(Neuron));
        if (net.layers[i]->neurons == NULL){
            free_network_contents(&net);
            free(data);
            return (Network){0};
        }
        read_model_bytes(data, file_size, &offset, net.layers[i]->neurons,
                         (size_t)neuron_counts[i] * sizeof(Neuron));
        for (int n = 0; n < neuron_counts[i]; n++) net.layers[i]->neurons[n].idx = n;
        read_model_bytes(data, file_size, &offset, &nconns, sizeof(int));
        net.layers[i]->n_conns = nconns;
        for (int nc = 0; nc < nconns; nc++){
            int source, rows;
            read_model_bytes(data, file_size, &offset, &source, sizeof(int));
            read_model_bytes(data, file_size, &offset, &rows, sizeof(int));
            net.layers[i]->conns[source] = calloc((size_t)rows, sizeof(float*));
            if (net.layers[i]->conns[source] == NULL){
                free_network_contents(&net);
                free(data);
                return (Network){0};
            }
            for (int row = 0; row < rows; row++){
                net.layers[i]->conns[source][row] = malloc((size_t)neuron_counts[i] * sizeof(float));
                if (net.layers[i]->conns[source][row] == NULL){
                    free_network_contents(&net);
                    free(data);
                    return (Network){0};
                }
                read_model_bytes(data, file_size, &offset,
                                 net.layers[i]->conns[source][row],
                                 (size_t)neuron_counts[i] * sizeof(float));
            }
        }
    }
    free(data);
    return net;
}

static int write_model_bytes(FILE* file, const void* data, size_t count, size_t size){
    return fwrite(data, size, count, file) == count;
}

int write_neural_bins(const char* filename, Network* net){
    if (filename == NULL || net == NULL || net->n_layers <= 0 || net->n_layers > MAX_LAYERS){
        fprintf(stderr, "Invalid network; model was not written\n");
        return 0;
    }
    size_t filename_length = strlen(filename);
    if (filename_length > SIZE_MAX - 32) return 0;
    char* temporary_filename = malloc(filename_length + 32);
    if (temporary_filename == NULL) return 0;
    snprintf(temporary_filename, filename_length + 32, "%s.tmp.%ld",
             filename, (long)getpid());
    FILE* fptr = fopen(temporary_filename, "wb");
    if (!fptr){
        perror("File could not be written");
        free(temporary_filename);
        return 0;
    }

    if (!write_model_bytes(fptr, &net->n_layers, 1, sizeof(int))) goto write_error;
    
    for (int i = 0; i < net->n_layers; i++){
        if (net->layers[i] == NULL || net->layers[i]->label == NULL
            || net->layers[i]->neurons == NULL || net->layers[i]->n_neurons <= 0){
            goto write_error;
        }
        int label_len = strlen(net->layers[i]->label);
        if (!write_model_bytes(fptr, &label_len, 1, sizeof(int))
            || !write_model_bytes(fptr, net->layers[i]->label, (size_t)label_len, sizeof(char))
            || !write_model_bytes(fptr, &net->layers[i]->n_neurons, 1, sizeof(int))
            || !write_model_bytes(fptr, &net->layers[i]->idx, 1, sizeof(int))
            || !write_model_bytes(fptr, net->layers[i]->neurons,
                                  (size_t)net->layers[i]->n_neurons, sizeof(Neuron))){
            goto write_error;
        }

        int nconns = 0;
        for (int k = 0; k < MAX_LAYERS; k++){
            if (net->layers[i]->conns[k]) nconns++;
        }
        net->layers[i]->n_conns = nconns;
        if (!write_model_bytes(fptr, &nconns, 1, sizeof(int))) goto write_error;

        for (int k = 0; k < MAX_LAYERS; k++){
            if (net->layers[i]->conns[k]){
                if (k >= net->n_layers || net->layers[k] == NULL) goto write_error;
                int rows = net->layers[k]->n_neurons;
                if (!write_model_bytes(fptr, &k, 1, sizeof(int))
                    || !write_model_bytes(fptr, &rows, 1, sizeof(int))) goto write_error;
                for(int l = 0; l < net->layers[k]->n_neurons; l++){
                    if (net->layers[i]->conns[k][l] == NULL
                        || !write_model_bytes(fptr, net->layers[i]->conns[k][l],
                                              (size_t)net->layers[i]->n_neurons, sizeof(float))){
                        goto write_error;
                    }
                }
            }
        }
    }
    
    if (fclose(fptr) != 0){
        perror("Error closing model file");
        remove(temporary_filename);
        free(temporary_filename);
        return 0;
    }
    if (rename(temporary_filename, filename) != 0){
        perror("Could not replace model file");
        remove(temporary_filename);
        free(temporary_filename);
        return 0;
    }
    free(temporary_filename);
    return 1;

write_error:
    fprintf(stderr, "Could not write complete model: %s\n", filename);
    fclose(fptr);
    remove(temporary_filename);
    free(temporary_filename);
    return 0;
}

#ifndef NEURA_NO_MAIN
static int find_named_layer(Network* net, const char* name){
    if (net == NULL || name == NULL) return -1;
    for (int i = 0; i < net->n_layers; i++){
        const char* label = net->layers[i] != NULL ? net->layers[i]->label : NULL;
        if (label == NULL) continue;
        if (strncasecmp(label, "LAYER_", 6) == 0) label += 6;
        if (strcasecmp(label, name) == 0) return i;
    }
    return -1;
}

static int restore_reversed_input_routes(Network* net, int input_layer){
    if (net == NULL || input_layer < 0 || input_layer >= net->n_layers) return 0;
    int outgoing_routes = 0;
    int reversed_routes = 0;
    for (int target = 0; target < net->n_layers; target++){
        if (target == input_layer || net->layers[target] == NULL) continue;
        if (net->layers[target]->conns[input_layer] != NULL) outgoing_routes++;
        if (net->layers[input_layer]->conns[target] != NULL) reversed_routes++;
    }
    if (outgoing_routes > 0 || reversed_routes == 0) return 0;

    float** restored[MAX_LAYERS] = {0};
    size_t input_width = (size_t)net->layers[input_layer]->n_neurons;
    for (int target = 0; target < net->n_layers; target++){
        if (target == input_layer || net->layers[target] == NULL
            || net->layers[input_layer]->conns[target] == NULL) continue;
        size_t target_width = (size_t)net->layers[target]->n_neurons;
        float** transposed = calloc(input_width, sizeof(float*));
        if (transposed == NULL) goto allocation_error;
        restored[target] = transposed;
        for (size_t input_neuron = 0; input_neuron < input_width; input_neuron++){
            transposed[input_neuron] = malloc(target_width * sizeof(float));
            if (transposed[input_neuron] == NULL) goto allocation_error;
            for (size_t target_neuron = 0; target_neuron < target_width; target_neuron++){
                transposed[input_neuron][target_neuron] =
                    net->layers[input_layer]->conns[target][target_neuron][input_neuron];
            }
        }
    }

    int restored_count = 0;
    for (int target = 0; target < net->n_layers; target++){
        if (restored[target] == NULL) continue;
        net->layers[target]->conns[input_layer] = restored[target];
        net->layers[target]->n_conns++;
        restored_count++;
    }
    printf("RESTORED_INPUT_ROUTES=%d\n", restored_count);
    return restored_count;

allocation_error:
    for (int target = 0; target < net->n_layers; target++){
        if (restored[target] == NULL) continue;
        for (size_t row = 0; row < input_width; row++) free(restored[target][row]);
        free(restored[target]);
    }
    return -1;
}

static int read_text_token(FILE* input, char** token, size_t* capacity,
                           size_t* token_length){
    size_t length = 0;
    int character;
    while ((character = fgetc(input)) != EOF && character != '\n'){
        if (length + 1 >= *capacity){
            if (*capacity > SIZE_MAX / 2) return -1;
            size_t new_capacity = *capacity * 2;
            char* resized = realloc(*token, new_capacity);
            if (resized == NULL) return -1;
            *token = resized;
            *capacity = new_capacity;
        }
        (*token)[length++] = (char)character;
    }
    if (character == EOF && ferror(input)) return -1;
    if (character == EOF && length == 0) return 0;
    if (length > 0 && (*token)[length - 1] == '\r') length--;
    (*token)[length] = '\0';
    *token_length = length;
    return 1;
}

static int process_text_stream(Network* net, FILE* input, size_t* processed_tokens,
                               int learn){
    int input_layer = find_named_layer(net, "INPUT");
    if (input_layer < 0) input_layer = 0;
    if (net == NULL || input == NULL || processed_tokens == NULL
        || net->n_layers <= 0 || net->layers[input_layer] == NULL
        || net->layers[input_layer]->n_neurons <= 0){
        return 0;
    }
    if (learn && restore_reversed_input_routes(net, input_layer) < 0) return 0;
    size_t width = (size_t)net->layers[input_layer]->n_neurons;
    if (width > SIZE_MAX / sizeof(float)) return 0;
    float* outputs = calloc(width, sizeof(float));
    float* timestamps = calloc(width, sizeof(float));
    size_t token_capacity = 256;
    char* token = malloc(token_capacity);
    if (outputs == NULL || timestamps == NULL || token == NULL){
        free(outputs);
        free(timestamps);
        free(token);
        return 0;
    }

    size_t count = 0;
    float timestamp = 0.0f;
    if (learn){
        for (int layer = 0; layer < net->n_layers; layer++){
            for (int neuron = 0; neuron < net->layers[layer]->n_neurons; neuron++){
                if (net->layers[layer]->neurons[neuron].current_timestamp > timestamp){
                    timestamp = net->layers[layer]->neurons[neuron].current_timestamp;
                }
            }
        }
        for (int layer = 0; layer < net->n_layers; layer++){
            for (int neuron = 0; neuron < net->layers[layer]->n_neurons; neuron++){
                net->layers[layer]->neurons[neuron].current_timestamp = timestamp;
            }
        }
    }
    int read_status;
    size_t token_length;
    while ((read_status = read_text_token(input, &token, &token_capacity,
                                          &token_length)) > 0){
        if (token_length == 0) continue;
        memset(outputs, 0, width * sizeof(float));
        timestamp += DT;
        for (size_t offset = 0; offset < token_length; offset++){
            uint64_t hash = UINT64_C(1469598103934665603);
            size_t fragment_length = token_length - offset;
            if (fragment_length > 3) fragment_length = 3;
            for (size_t byte = 0; byte < fragment_length; byte++){
                hash ^= (unsigned char)token[offset + byte];
                hash *= UINT64_C(1099511628211);
            }
            outputs[hash % width] = 70.0f;
        }
        for (size_t neuron = 0; neuron < width; neuron++){
            timestamps[neuron] = timestamp - DT * 0.5f;
            if (outputs[neuron] > 0.0f){
                Data input_spike = {(int)neuron, input_layer, outputs[neuron], timestamp};
                logger_log(EVENT_TYPE_SPIKE, input_spike);
            }
        }
        LayerSignal input_signal = {
            .outputs = outputs,
            .spike_timestamps = timestamps,
            .n_outputs = net->layers[input_layer]->n_neurons,
            .from = input_layer,
        };
        if (learn) signal(net, &input_signal, 0);
        else signal_inference(net, &input_signal, 1);
        count++;
    }
    free(outputs);
    free(timestamps);
    free(token);
    if (read_status < 0) return 0;
    *processed_tokens = count;
    return 1;
}

int main(int argc, char** argv){
    char* create_filename = NULL;
    char* save_filename = NULL;
    char* read_filename = NULL;
    int label_true = 0;
    int debug = 0;
    int train_text = 0;
    int infer_text = 0;
    char* log_filename = NULL;

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
        } else if (strcmp(argv[i], "--train-text") == 0){
            train_text = 1;
        } else if (strcmp(argv[i], "--infer-text") == 0){
            infer_text = 1;
        } else if (strcmp(argv[i], "--log") == 0 && i != argc-1){
            log_filename = argv[i+1];
        } else if (strcmp(argv[i], "--audio") == 0){
            record_audio = 1;
        }
    }

    if ((train_text && infer_text) || ((train_text || infer_text) && record_audio)){
        fprintf(stderr, "Text training/inference and audio are mutually exclusive\n");
        return 1;
    }
    if (infer_text && (read_filename == NULL || save_filename != NULL)){
        fprintf(stderr, "--infer-text requires --rfile and does not accept --sfile\n");
        return 1;
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
            Network loaded = read_neural_bins(read_filename);
            if (loaded.n_layers <= 0){
                return 1;
            }
            l = malloc(sizeof(Network));
            if (l == NULL){
                free_network_contents(&loaded);
                return 1;
            }
            *l = loaded;
        }
        if (debug && l != NULL){
            printf("DEBUG: Layer model read\n");

            for (int layer = 0; layer < l->n_layers; layer++){
                for (int i = 0; i < MAX_LAYERS; i++){
                    if (l->layers[layer]->conns[i] != NULL){
                        if (i < l->n_layers && l->layers[i]->n_neurons > 1
                            && l->layers[layer]->n_neurons > 1){
                            printf("SAMPLE: %f\n", l->layers[layer]->conns[i][1][1]);
                        }
                        printf("LAYER %d to LAYER %d - OK\n", layer, i);
                    }
                }            
            }
            sleep(2);
        }

        if (train_text || infer_text){
            if (train_text && save_filename == NULL){
                fprintf(stderr, "--train-text requires --sfile\n");
                free_network(l);
                return 1;
            }
            if (log_filename != NULL) init_logger(log_filename);
            logger_reset_counts();
            size_t processed_tokens = 0;
            int processed = process_text_stream(l, stdin, &processed_tokens, train_text);
            logger_end();
            if (!processed || processed_tokens == 0){
                fprintf(stderr, "No text tokens were processed or input failed\n");
                free_network(l);
                return 1;
            }
            unsigned long spike_count = 0;
            unsigned long weight_updates = 0;
            unsigned long spikes_by_layer[MAX_LAYERS] = {0};
            logger_get_counts(&spike_count, &weight_updates, spikes_by_layer,
                              (size_t)l->n_layers);
            printf("%s_TOKENS=%zu\n", train_text ? "TRAINED" : "INFERRED", processed_tokens);
            printf("SPIKE_COUNT=%lu\n", spike_count);
            printf("WEIGHT_UPDATES=%lu\n", weight_updates);
            printf("SPIKES_BY_LAYER=");
            for (int layer = 0; layer < l->n_layers; layer++){
                printf("%s%lu", layer == 0 ? "" : ",", spikes_by_layer[layer]);
            }
            putchar('\n');
            if (train_text && !write_neural_bins(save_filename, l)){
                free_network(l);
                return 1;
            }
        } else if (record_audio) {
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
        if (save_filename && !train_text && !infer_text){
            if (!write_neural_bins(save_filename, l)){
                free_network(l);
                return 1;
            }
        }
        free_network(l);
    } else {
        printf("Error, --rfile xor --sfile => 0\n");
        return 1;
    }
    return 0;
}
#endif

