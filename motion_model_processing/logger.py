import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = None

    def start_logging(self, file_name_prefix="training"):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{file_name_prefix}_{date_str}.txt"
        self.log_file = os.path.join(self.log_dir, file_name)

        with open(self.log_file, 'w') as f:
            f.write(f"Logging started at {datetime.now()}\n")

        return self.log_file

    def log_hyperparameters(self, params_dict):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write("\nHyperparameters:\n")
                for key, value in params_dict.items():
                    f.write(f"{key}: {value}\n")

    def log_training_output(self, message):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")

    def end_logging(self):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"\nLogging ended at {datetime.now()}\n")
        print(f"Training log saved to {self.log_file}")
