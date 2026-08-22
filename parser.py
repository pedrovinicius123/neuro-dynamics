import struct
import sys
import json
from enum import IntEnum
from dataclasses import dataclass
from typing import List

class Event(IntEnum):
    EVENT_TYPE_SPIKE = 0
    EVENT_TYPE_WEIGTH_UPDATE = 1

@dataclass
class Data:
    neuron_idx: int
    layer_idx: int
    neuron_u: float
    timestamp: float
    
    @staticmethod
    def from_bytes(data: bytes) -> 'Data':
        # Formato: i i f f (int, int, float, float)
        # i = signed int (4 bytes)
        # f = float (4 bytes)
        neuron_idx, layer_idx, neuron_u, timestamp = struct.unpack('<iiff', data)
        return Data(neuron_idx, layer_idx, neuron_u, timestamp)

@dataclass
class LogEntry:
    event_type: Event
    data: Data
    
    @staticmethod
    def from_bytes(data: bytes) -> 'LogEntry':
        # Primeiro lê o event_type (int)
        event_type_value = struct.unpack('<i', data[:4])[0]
        event_type = Event(event_type_value)
        
        # Depois lê a Data (16 bytes: 2 ints + 2 floats)
        data_bytes = data[4:20]
        data_obj = Data.from_bytes(data_bytes)
        
        return LogEntry(event_type, data_obj)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.name,
            "neuron_idx": self.data.neuron_idx,
            "layer_idx": self.data.layer_idx,
            "neuron_u": self.data.neuron_u,
            "timestamp": self.data.timestamp,
        }

def read_log_file(filename: str) -> List[LogEntry]:
    """Lê todas as entradas do arquivo binário"""
    entries = []
    entry_size = 20  # 4 bytes (event_type) + 16 bytes (Data)
    
    try:
        with open(filename, 'rb') as f:
            while True:
                chunk = f.read(entry_size)
                if not chunk:
                    break
                if len(chunk) < entry_size:
                    print(f"Warning: Arquivo truncado, ignorando {len(chunk)} bytes")
                    break
                
                entry = LogEntry.from_bytes(chunk)
                entries.append(entry)
                
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filename}' não encontrado")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        sys.exit(1)
    
    return entries

def print_entries(entries: List[LogEntry], verbose: bool = False):
    """Exibe as entradas de forma legível"""
    print(f"\nTotal de entradas: {len(entries)}\n")
    
    for i, entry in enumerate(entries):
        event_name = entry.event_type.name
        
        if verbose or entry.event_type == Event.EVENT_TYPE_SPIKE:
            print(f"Entry {i:4d}: {event_name:20s} | "
                  f"Layer: {entry.data.layer_idx:3d} | "
                  f"Neuron: {entry.data.neuron_idx:4d} | "
                  f"U: {entry.data.neuron_u:8.4f} | "
                  f"Timestamp: {entry.data.timestamp:10.4f}")
        else:
            print(f"Entry {i:4d}: {event_name:20s} | "
                  f"Weight Update | "
                  f"Layer: {entry.data.layer_idx:3d} | "
                  f"Neuron: {entry.data.neuron_idx:4d} | "
                  f"Value: {entry.data.neuron_u:8.4f} | "
                  f"Timestamp: {entry.data.timestamp:10.4f}")

def analyze_entries(entries: List[LogEntry]):
    """Análise estatística básica das entradas"""
    spike_count = 0
    weight_update_count = 0
    layers = {}
    timestamps = []
    
    for entry in entries:
        if entry.event_type == Event.EVENT_TYPE_SPIKE:
            spike_count += 1
        else:
            weight_update_count += 1
        
        layer = entry.data.layer_idx
        if layer not in layers:
            layers[layer] = 0
        layers[layer] += 1
        
        timestamps.append(entry.data.timestamp)
    
    print("\n=== ANÁLISE ESTATÍSTICA ===")
    print(f"Total spikes: {spike_count}")
    print(f"Total weight updates: {weight_update_count}")
    print(f"Proporção spikes/updates: {spike_count/weight_update_count:.2f}" if weight_update_count > 0 else "N/A")
    
    print(f"\nEntradas por camada:")
    for layer in sorted(layers.keys()):
        print(f"  Layer {layer}: {layers[layer]} entradas")
    
    if timestamps:
        print(f"\nTimestamps:")
        print(f"  Mínimo: {min(timestamps):.4f}")
        print(f"  Máximo: {max(timestamps):.4f}")
        print(f"  Duração total: {max(timestamps) - min(timestamps):.4f}")

def export_to_csv(entries: List[LogEntry], filename: str):
    """Exporta os dados para CSV"""
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['index', 'event_type', 'neuron_idx', 'layer_idx', 'neuron_u', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, entry in enumerate(entries):
            writer.writerow({
                'index': i,
                'event_type': entry.event_type.name,
                'neuron_idx': entry.data.neuron_idx,
                'layer_idx': entry.data.layer_idx,
                'neuron_u': entry.data.neuron_u,
                'timestamp': entry.data.timestamp
            })
    
    print(f"\nDados exportados para {filename}")

def export_to_json(entries: List[LogEntry], filename: str):
    """Exporta eventos em um formato legível por ferramentas externas."""
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump([entry.to_dict() for entry in entries], jsonfile, indent=2)
    print(f"\nDados exportados para {filename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Leitor de arquivos binários de log de rede neural')
    parser.add_argument('filename', help='Caminho do arquivo binário')
    parser.add_argument('-v', '--verbose', action='store_true', help='Mostra todas as entradas')
    parser.add_argument('-a', '--analyze', action='store_true', help='Mostra análise estatística')
    parser.add_argument('-c', '--csv', help='Exporta para CSV (forneça o nome do arquivo)')
    parser.add_argument('-j', '--json', help='Exporta eventos para JSON')
    parser.add_argument('-n', '--num-entries', type=int, help='Número de entradas para mostrar')
    
    args = parser.parse_args()
    
    # Lê o arquivo
    entries = read_log_file(args.filename)
    
    if not entries:
        print("Nenhuma entrada encontrada no arquivo")
        return
    
    # Mostra as entradas
    num_to_show = args.num_entries if args.num_entries else min(20, len(entries))
    print_entries(entries[:num_to_show], args.verbose)
    
    if len(entries) > num_to_show:
        print(f"\n... e mais {len(entries) - num_to_show} entradas")
    
    # Análise estatística
    if args.analyze:
        analyze_entries(entries)
    
    # Exporta para CSV
    if args.csv:
        export_to_csv(entries, args.csv)
    if args.json:
        export_to_json(entries, args.json)

if __name__ == "__main__":
    # Exemplo de uso direto (sem argumentos de linha de comando)
    if len(sys.argv) == 1:
        # Configuração padrão para teste
        filename = "logs/log.1.neur"
        print(f"Lendo arquivo: {filename}")
        
        try:
            entries = read_log_file(filename)
            print_entries(entries[:10], verbose=True)
            analyze_entries(entries)
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado. Use argumentos:")
            print("  python leitor_log.py <arquivo.bin> [-v] [-a] [-c saida.csv]")
    else:
        main()
