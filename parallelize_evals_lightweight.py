import subprocess
import queue
import threading
import time
import os
import sys
import json
import argparse
import itertools
from datetime import datetime
from task import TASK_MAPPING


class GPUJobQueue:
    def __init__(self, num_gpus=8, log_dir="job_logs"):
        self.num_gpus = num_gpus
        self.job_queue = queue.Queue()
        self.gpu_locks = [threading.Lock() for _ in range(num_gpus)]
        self.running_processes = [None] * num_gpus
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.queue_file = os.path.join(self.log_dir, "queued_commands.json")
        self.completed_file = os.path.join(self.log_dir, "completed_commands.json")
        self.log_files = [
            os.path.join(self.log_dir, f"gpu{i}.log") for i in range(num_gpus)
        ]
        self.queue_lock = threading.Lock()

        # Intialize completed jobs with empty list
        with open(self.completed_file, "w") as f:
            json.dump([], f, indent=4)

    def _save_queue(self):
        with self.queue_lock:
            try:
                with open(self.queue_file, "w") as f:
                    json.dump(list(self.job_queue.queue), f, indent=4)
            except Exception as e:
                print(f"Error saving queue to {self.queue_file}: {str(e)}")

    def _save_completed(self, command):
        with self.queue_lock:
            try:
                with open(self.completed_file, "r+") as f:
                    completed = json.load(f)
                    completed.append(command)
                    f.seek(0)
                    json.dump(completed, f, indent=4)
                    f.truncate()
            except Exception as e:
                print(f"Error updating {self.completed_file}: {str(e)}")

    def add_job(self, bash_command):
        self.job_queue.put(bash_command)
        self._save_queue()

    def run_job(self, gpu_id, bash_command):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_files[gpu_id]

        try:
            with open(log_file, "a") as log:
                log.write(f"Running command: {bash_command}\n")
                log.write(f"GPU: {gpu_id}\n")
                log.write(
                    f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                log.write("-" * 50 + "\n")
                log.flush()

                process = subprocess.Popen(
                    bash_command,
                    shell=True,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                self.running_processes[gpu_id] = process

                process.wait()

                log.write("\n" + "-" * 50 + "\n")
                log.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write(f"Exit code: {process.returncode}\n")

            self._save_completed(bash_command)
        except Exception as e:
            print(f"Error running job on GPU {gpu_id}: {str(e)}")
        finally:
            self.gpu_locks[gpu_id].release()
            self.running_processes[gpu_id] = None

    def process_queue(self):
        while True:
            if self.job_queue.empty() and all(
                proc is None for proc in self.running_processes
            ):
                break

            for gpu_id in range(self.num_gpus):
                if self.running_processes[gpu_id] is None and self.gpu_locks[
                    gpu_id
                ].acquire(blocking=False):
                    if not self.job_queue.empty():
                        bash_command = self.job_queue.get()
                        threading.Thread(
                            target=self.run_job, args=(gpu_id, bash_command)
                        ).start()
                        self._save_queue()
                    else:
                        self.gpu_locks[gpu_id].release()

            time.sleep(1)  # Small delay to prevent busy-waiting

    def terminate_all_jobs(self):
        print("Terminating all running jobs...")
        for gpu_id, process in enumerate(self.running_processes):
            if process is not None:
                process.terminate()
                with open(self.log_files[gpu_id], "a") as log:
                    log.write(
                        f"\nJob terminated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    log.write("=" * 50 + "\n\n")
        print("All jobs terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run eval jobs for given lightweight model version"
    )
    parser.add_argument(
        "--lightweight_model_versions",
        type=str,
        nargs="+",
        required=True,
        help="List of lightweight model versions to be evaluated.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required="--command_file" not in sys.argv,
        choices=list(TASK_MAPPING.keys()),
        help="List of tasks to be evaluated.",
    )
    parser.add_argument(
        "--cache_sizes",
        type=float,
        nargs="+",
        default=[8192, 4096, 2048, 1024, 512, 256, 128],
        help="Cache sizes to be evaluated.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of examples to sample for evaluation. Defaults to None, which uses the full dataset.",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Number of GPUs available"
    )
    parser.add_argument(
        "--log_dir", default="eval_job_logs", help="Directory for job logs"
    )

    args = parser.parse_args()

    gpu_queue = GPUJobQueue(num_gpus=args.num_gpus, log_dir=args.log_dir)

    base_command = "python eval.py --task {task} --cache_strategy lightweight --prompt_compression_strategy lightweight --lightweight_model_version {version} --global_tokens 4 --recent_window 10 --num_samples {ns}  --max_cache_length {cs} --use_wandb"  # --compile

    for task, cs, version in itertools.product(
        args.tasks, args.cache_sizes, args.lightweight_model_versions
    ):
        gpu_queue.add_job(
            base_command.format(
                task=task,
                version=version,
                ns=args.num_samples,
                cs=cs,
            )
        )

    print(f"Adding {gpu_queue.job_queue.qsize()} tasks into the job queue")

    try:
        gpu_queue.process_queue()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, terminating all jobs...")
        gpu_queue.terminate_all_jobs()
        print("Exiting.")

    print("All jobs completed or terminated")
