import numpy as np
import subprocess
from datetime import datetime

class GPUMonitor:
    def __init__(self, gpu_ids=[0], interval=1):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.log_file = f"results/gpu_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.process = None
        
    def start(self):
        with open(self.log_file, 'w') as f:
            f.write("timestamp,gpu_id,utilization\n")

        cmd = f"""
        while true; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu \
                      --format=csv,noheader,nounits \
                      -i {','.join(map(str, self.gpu_ids))} >> {self.log_file}
            sleep {self.interval}
        done
        """
        
        self.process = subprocess.Popen(cmd, shell=True)
        print(f"Started GPU monitoring, logging to {self.log_file}")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Stopped GPU monitoring")
            
            try:
                with open(self.log_file, 'r') as f:
                    next(f)
                    
                    gpu_utils = {}
                    for line in f:
                        _, gpu_id, util = line.strip().split(',')
                        if gpu_id not in gpu_utils:
                            gpu_utils[gpu_id] = []
                        gpu_utils[gpu_id].append(float(util))
                
                print("\nGPU Utilization Summary:")
                for gpu_id, utils in gpu_utils.items():
                    avg_util = sum(utils) / len(utils)
                    max_util = max(utils)
                    median_util = np.median(utils)
                    print(f"GPU {gpu_id}:")
                    print(f"  Average: {avg_util:.1f}%")
                    print(f"  Median:  {median_util:.1f}%")
                    print(f"  Maximum: {max_util:.1f}%")
            except Exception as e:
                print(f"Error generating summary: {e}")