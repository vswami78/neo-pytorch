import argparse
import logging
from logging.handlers import RotatingFileHandler
import sys

from config_manager import ConfigManager
from core.graph_builder import create_heterogeneous_graph
from interactive_graph_explorer import InteractiveGraphExplorer
from analysis import run_analysis

class GraphExplorerApp:
    def __init__(self, config_path):
        self.logger = self._setup_logger()
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_and_validate()
        self.hetero_data = None

    def _setup_logger(self):
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

    def build_graph(self):
        """
        Builds the heterogeneous graph based on the loaded configuration.
        This step is crucial for preparing the data structure for exploration.
        """
        try:
            self.hetero_data = create_heterogeneous_graph(
                self.config.input_file,
                self.config.node_types,
                self.config.edge_types,
                self.logger
            )
            self.logger.info("Graph built successfully")
        except Exception as e:
            self.logger.error(f"Failed to build graph: {e}")
            sys.exit(1)

    def run_explorer(self):
        """
        Launches the interactive graph explorer if the graph data is available.
        This method provides the main user interface for graph exploration.
        """
        if not self.hetero_data:
            self.logger.error("Graph data not built. Call build_graph() first.")
            return

        try:
            explorer = InteractiveGraphExplorer(
                self.hetero_data,
                self.logger,
                self.config  # Pass the entire config object
            )
            explorer.run()
        except Exception as e:
            self.logger.error(f"Error during graph exploration: {e}")

    def run_analysis(self):
        """
        Performs analysis on the graph data.
        This method can be expanded to include various types of graph analysis.
        """
        if not self.hetero_data:
            self.logger.error("Graph data not built. Call build_graph() first.")
            return

        try:
            results = run_analysis(self.hetero_data, self.logger)
            self.logger.info("Analysis completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Graph Explorer Application")
    parser.add_argument("--config", default="config/graph_config_v2.json", help="Path to the configuration file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    app = GraphExplorerApp(args.config)
    app.build_graph()
    app.run_explorer()
    # Uncomment the following line to run analysis
    # analysis_results = app.run_analysis()
