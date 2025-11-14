# Assumes a Redis server is running at localhost:6379

import redis
import glob
import os

DATA_DIR = "/data/group_data/rl/craftax_unlabelled_new/"
FILE_PATTERN = "trajectories_batch_*.npz"
QUEUE_NAME = "craftax_job_queue"
RESULTS_DIR = "/data/group_data/rl/craftax_labelled_results/"


# This assumes default localhost:6379, change if needed
r = redis.Redis(host='login2', port=6379, decode_responses=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
file_paths = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
file_paths.sort()

with r.pipeline() as pipe:
    for path in file_paths[-610:]:
        pipe.lpush(QUEUE_NAME, path)
    pipe.execute()

print(f"Pushed {len(file_paths[-610:])} jobs to '{QUEUE_NAME}'.")
