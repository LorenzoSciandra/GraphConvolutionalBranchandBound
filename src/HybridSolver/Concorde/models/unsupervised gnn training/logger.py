import logging
from typing import Dict
import wandb
import os
import json
import hashlib
from typing import Dict, Union


def generate_config_hash(config: dict) -> str:
    """Generates a unique hash for a given configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True)  # Convert config to JSON string for hashing
    return hashlib.md5(config_str.encode()).hexdigest()  # Generate MD5 hash

class Logger:
    def __init__(self, name: str, logs_directory: str = './logs/', results_directory: str = './results/'):
        self.logger = logging.getLogger(name)
        self.name = name
        log_file_path = os.path.join(logs_directory, f"{name}.log")

        if os.path.exists(log_file_path):
            print(f"Logger with configuration {self.name} already exists. Reusing it.")
        self.logs_directory = logs_directory
        self.results_directory = results_directory
        self._configure_local_logger(name)
        
    def _configure_local_logger(self, name: str):
        """Configures the local logger with console and file handlers."""
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # File handler
        file_handler = logging.FileHandler(f"{self.logs_directory}{name}.log", mode='w', encoding="utf-8")
        
        # Formatter
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M"
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)

    def __call__(self, stats: Dict[str, float]):
        """Logs the statistics both to the console and file."""
        message = " | ".join(f"{key}: {value}" for key, value in stats.items())
        self.logger.info(message)
    
    def finish(self):
        self.logger.info('finish')

class WandbLogger(Logger):
    def __init__(self, name: str, logs_directory: str, results_directory: str, **kwargs):
        super().__init__(name, logs_directory, results_directory)
        # self.project_name = project_name
        # Initialize the Wandb run
        wandb.init(name=name, **kwargs)
        
    def __call__(self, stats: Dict[str, float]):
        """Logs statistics to both the local logger and Wandb."""
        super().__call__(stats)  # Log to local logger
        wandb.log(stats, step = stats['step'])         # Log to Wandb

    def finish(self):
        """Ends the Wandb run."""
        super().finish()
        wandb.finish()


# Assuming Logger and WandbLogger classes are imported here

def initialize_logger_from_config(logger_config: dict) -> Union[Logger, WandbLogger]:
    
    print(logger_config)
    logger_type = logger_config.get("type", "local").lower()
    logs_directory = "./logs/"
    results_directory = "./checkpoints/"
    os.makedirs(logs_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)

    # Generate a unique hash for the logger configuration
    config_hash = generate_config_hash(logger_config)
    name = f"{config_hash}"
    
    # Initialize either Logger or WandbLogger based on configuration
    if logger_type == "wandb":
        
        # Gather additional Wandb arguments from config
        kwargs = logger_config.get("wandb_args", [])
        print(f"kwargs: {kwargs}")
        # Filter out any None values in kwargs
        # merge all the dict values in kwargs in one dict
        kwargs = {k: v for d in kwargs for k, v in d.items() if v is not None}
        
        print(f"kwargs: {kwargs}")
        
        return WandbLogger(name=name, logs_directory=logs_directory, results_directory=results_directory, **kwargs)
    else:
        return Logger(name=name, logs_directory=logs_directory, results_directory=results_directory)