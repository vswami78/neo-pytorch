import logging
from logging.handlers import RotatingFileHandler
import sys
import random
import torch

from config_manager import ConfigManager
from core.graph_builder import create_heterogeneous_graph
from interactive_graph_explorer import InteractiveGraphExplorer
# from analysis import run_analysis
from graph_analysis import analyze_graph, generate_analysis_report


def setup_logging():
    log_file = 'graph_explorer.log'
    logger = logging.getLogger("graph_explorer")
    logger.setLevel(logging.ERROR)  # Set this to control your application's logging level
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=5)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger


def sample_ip_cases(hetero_data, config, num_samples, num_hops):
    ip_identifier = config.node_types['IP']
    ip_mapping = hetero_data['IP'][ip_identifier]

    edge_index = hetero_data[('IP', 'belongs_to', 'user')].edge_index
    unique_ips, counts = edge_index[0].unique(return_counts=True)
    
    filtered_ips = unique_ips[(counts >= num_samples) & (counts <= num_hops)]
    
    if len(filtered_ips) > 10:
        sampled_ips = random.sample(filtered_ips.tolist(), 10)
    else:
        sampled_ips = filtered_ips.tolist()
    
    samples = []
    for ip_index in sampled_ips:
        num_cases = counts[ip_index].item()
        ip_value = ip_mapping[ip_index]
        
        # Convert to Python native type if it's a tensor
        if isinstance(ip_value, torch.Tensor):
            ip_value = ip_value.item()
        
        # Convert to string for consistency in output
        ip_str = str(ip_value)
        
        samples.append((ip_value, num_cases))

    return samples

def sample_device_cases(hetero_data, config, min_users, max_users):
    device_identifier = config.node_types['device']
    
    # Get the device-user relationships
    edge_index = hetero_data[('device', 'belongs_to', 'user')].edge_index
    
    # Count the number of unique users per device
    unique_devices, inverse_indices = edge_index[0].unique(return_inverse=True)
    unique_users = edge_index[1].unique()
    
    device_to_users = {device.item(): set() for device in unique_devices}
    for device_idx, user_idx in zip(inverse_indices, edge_index[1]):
        device_to_users[unique_devices[device_idx].item()].add(user_idx.item())
    
    # Filter devices with unique user count between min_users and max_users
    filtered_devices = [(device, len(users)) for device, users in device_to_users.items() 
                        if min_users < len(users) <= max_users]
    
    # Sample up to 10 devices
    sampled_devices = random.sample(filtered_devices, min(10, len(filtered_devices)))
    
    # Get the original device IDs
    original_device_ids = hetero_data['device'].original_ids
    
    # Prepare the results
    results = []
    for device_index, user_count in sampled_devices:
        original_device_id = original_device_ids[device_index]
        results.append((original_device_id, user_count))
    
    return results

if __name__ == "__main__":
    logger = setup_logging()
    config_path = "config/graph_config_v3.json"

    config_manager = ConfigManager(config_path)
    config = config_manager.load_and_validate()

    hetero_data = create_heterogeneous_graph(config, logger)
    
    # Perform graph analysis
    analysis_results = analyze_graph(hetero_data)
    analysis_report = generate_analysis_report(analysis_results)
    
    # Print the analysis report
    # print(analysis_report)
    
    # # Sample and print cases for IPs belonging to multiple users
    # ip_samples = sample_ip_cases(hetero_data, config, 1, 10)
    # print("\nSample cases where 1 < IP belongs to <= 10 users:")
    # for ip, count in ip_samples:
    #     print(f"IP: {ip}, Belongs to {count} users")

    # Sample and print cases for devices belonging to multiple users
    device_samples = sample_device_cases(hetero_data, config, 1, 10)
    print("\nSample cases where 1 < device belongs to <= 10 users:")
    for device, count in device_samples:
        print(f"Device: {device}, Belongs to {count} users")

    # Optionally, save the report to a file
    with open("graph_analysis_report.txt", "w") as f:
        f.write(analysis_report)

    # Create and run the InteractiveGraphExplorer
    explorer = InteractiveGraphExplorer(hetero_data, config, logger)
    explorer.run()
