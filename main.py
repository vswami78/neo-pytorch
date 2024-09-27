import logging
from logging.handlers import RotatingFileHandler
import sys

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
    print(analysis_report)
    
    # Optionally, save the report to a file
    with open("graph_analysis_report.txt", "w") as f:
        f.write(analysis_report)

    # Create and run the InteractiveGraphExplorer
    explorer = InteractiveGraphExplorer(hetero_data, config, logger)
    explorer.run()
