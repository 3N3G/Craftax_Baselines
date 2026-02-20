"""
LLM Worker for Craftax Data Labelling

Processes trajectory data through vLLM server to extract last-token hidden state
representations from game observations.

Supports two modes (controlled by GENERATE_TEXT flag):
1. Direct extraction (GENERATE_TEXT=False, default):
   - Extracts last-token hidden states directly from prompts
   - ~34x faster, no text generation
   - Suitable for training policies

2. Generation mode (GENERATE_TEXT=True):
   - Generates text reasoning first, then takes last-token hidden state
   - Slower but provides both text and hidden states
   - Useful for analysis or when text is needed

Hidden state output format: (N, hidden_size) — last-token hidden state only.
See docs/progress_journal.md for rationale.

Requires vLLM server running:
  bash scripts/start_vllm_hidden.sh --mode last_token
"""

import redis
import numpy as np
import os
import time
import logging
import socket
import wandb
import sys
import json
import requests
from obs_to_text import obs_to_text  # Decode symbolic observations to text

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_extractor import VLLMHiddenStateExtractor
from utils.llm_prompts import filter_text_obs

# --- Constants ---
QUEUE_NAME = "craftax_llm_job_queue"  # Separate queue name to avoid conflicts
RESULTS_DIR = "/data/group_data/rl/geney/vllm_craftax_labelled_results/"
LOGS_DIR = "/data/group_data/rl/geney/craftax_llm_job_logs/"
TEMP_NPY_DIR = os.path.join(RESULTS_DIR, "temp_npy")
PROGRESS_DIR = os.path.join(RESULTS_DIR, "progress")

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
BATCH_SIZE = 16  # 32 OOMed, 16 should be safe with ~50% headroom
TOKENS_GENERATED = 256  # Token budget for thinking + answer
GENERATE_TEXT = False  # Set to True to generate text before extracting hidden states

# --- mmap/save Constants ---
MAX_TEXT_LEN = 2048
TEXT_DTYPE = f'<U{MAX_TEXT_LEN}'
# Hidden states are saved as (N, hidden_size) — last-token only
# This matches what the policy network consumes and is ~256x smaller than per-token

# --- Standard Logging Setup ---
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_NPY_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)
pid = os.getpid()
hostname = socket.gethostname()
log_filename = os.path.join(LOGS_DIR, f"worker_{hostname}_{pid}.log")
logger = logging.getLogger(f"worker_{pid}")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_filename)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

# --- Redis Connection ---
# Read hostname from coordinator's shared file
REDIS_HOST_FILE = "/data/group_data/rl/geney/redis_host.txt"
try:
    with open(REDIS_HOST_FILE, 'r') as f:
        REDIS_HOST = f.read().strip()
    logger.info(f"Read Redis host from file: {REDIS_HOST}")
except FileNotFoundError:
    REDIS_HOST = "login1"  # Fallback
    logger.warning(f"Redis host file not found, using fallback: {REDIS_HOST}")

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
r.ping()
logger.info(f"Successfully connected to Redis at {REDIS_HOST}:6379")

# --- vLLM Server Check ---
VLLM_URL = "http://localhost:8000"
logger.info(f"Checking for vLLM server at {VLLM_URL}...")
try:
    resp = requests.get(f"{VLLM_URL}/health", timeout=5)
    if resp.status_code != 200:
        raise Exception(f"Server returned status {resp.status_code}")
    logger.info(f"✅ vLLM server ready at {VLLM_URL}")
except Exception as e:
    logger.error(f"❌ vLLM server not available: {e}")
    logger.error(f"Please start the server first:")
    logger.error(f"  vllm serve configs/vllm_hidden_last --max-model-len 8192 --gpu-memory-utilization 0.95 \\")
    logger.error(f"    --kv-transfer-config '{{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{{\"shared_storage_path\":\"/tmp/hidden_states\",\"mode\":\"last_token\"}}}}'")
    sys.exit(1)

# --- Initialize vLLM Extractor ---
logger.info(f"Initializing VLLMHiddenStateExtractor...")
# Use the model that the server actually loads
model_name = "./configs/vllm_hidden_qwen4b"
target_layer = -1  # Last of 4 extracted layers (layer 35)

extractor = VLLMHiddenStateExtractor(
    server_url=VLLM_URL,
    model_name=model_name,
    model_id=MODEL_ID,  # For tokenizer
    target_layer=target_layer,
    max_workers=BATCH_SIZE,  # Concurrent requests
)
logger.info(f"VLLMHiddenStateExtractor initialized.")

HIDDEN_SIZE = extractor.hidden_size
logger.info(f"Hidden size: {HIDDEN_SIZE}")




def write_progress(progress_path, batch_idx):
    """Atomically writes the last completed batch index."""
    with open(progress_path, 'w') as f:
        json.dump({"last_completed_batch": batch_idx}, f)

def read_progress(progress_path):
    """Reads the progress file. Returns -1 if not found."""
    if not os.path.exists(progress_path):
        return -1
    try:
        with open(progress_path, 'r') as f:
            data = json.load(f)
            return data.get("last_completed_batch", -1)
    except (json.JSONDecodeError, IOError):
        logger.warning(f"Could not read progress file {progress_path}. Starting from scratch.")
        return -1

# --- Main Worker Loop ---
while True:
    file_path = None
    try:
        # 1. GET JOB
        file_path = r.rpop(QUEUE_NAME)
        if file_path is None:
            logger.info("No more jobs! Exiting.")
            break

        logger.info(f"Processing job: {file_path}")
        job_basename = os.path.basename(file_path)
        wandb.init(project="craftax_offline_llm_labelling", name=f"labelling_{job_basename}", resume="allow")

        # Define paths for mmap "save state" files
        temp_hidden_path = os.path.join(TEMP_NPY_DIR, f"{job_basename}_temp_hidden.npy")
        temp_text_path = os.path.join(TEMP_NPY_DIR, f"{job_basename}_temp_text.npy")
        progress_path = os.path.join(PROGRESS_DIR, f"{job_basename}_progress.json")
        
        data = np.load(file_path, allow_pickle=True)
        num_samples = len(data["obs"])

        # 2. CHECK FOR SAVED PROGRESS
        last_completed_batch = read_progress(progress_path)
        start_batch = last_completed_batch + 1
        start_index = start_batch * BATCH_SIZE
        
        # Hidden states saved as (N, hidden_size) — last-token only
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch} (sample index {start_index})")
            hidden_states_memmap = np.memmap(
                temp_hidden_path, dtype=np.float16, mode='r+',
                shape=(num_samples, HIDDEN_SIZE)
            )
            text_outputs_memmap = np.memmap(
                temp_text_path, dtype=TEXT_DTYPE, mode='r+', shape=(num_samples,)
            )
        else:
            logger.info("Starting new job, creating temp files.")
            hidden_states_memmap = np.memmap(
                temp_hidden_path, dtype=np.float16, mode='w+',
                shape=(num_samples, HIDDEN_SIZE)
            )
            text_outputs_memmap = np.memmap(
                temp_text_path, dtype=TEXT_DTYPE, mode='w+', shape=(num_samples,)
            )
            write_progress(progress_path, -1)

        # 3. RUN INFERENCE (THE LONG PART)
        logger.info(f"Beginning inference from index {start_index}...")
        start_time = time.time()

        for i in range(start_index, num_samples, BATCH_SIZE):
            current_batch_idx = i // BATCH_SIZE
            current_batch_indices = range(i, min(i + BATCH_SIZE, num_samples))
            current_batch_size = len(current_batch_indices)

            # Collect filtered observations for this batch
            batch_observations = []
            for idx in current_batch_indices:
                # Use pre-saved text_obs if available, otherwise decode from obs
                if "text_obs" in data and data["text_obs"][idx]:
                    raw_text_obs = str(data["text_obs"][idx])
                else:
                    # Decode symbolic observation to text
                    raw_text_obs = obs_to_text(data["obs"][idx])

                # Filter to show only interesting tiles (remove background)
                filtered_text_obs = filter_text_obs(raw_text_obs)
                batch_observations.append(filtered_text_obs)

            if GENERATE_TEXT:
                # Mode 1: Generate text first, then take last-token hidden state
                # Slower but provides both text and hidden states
                batch_hidden_vectors, generated_texts, metrics = extractor.extract_hidden_states(
                    batch_observations,
                    batch_size=BATCH_SIZE
                )
                # extract_hidden_states returns (N, hidden_size) last-token hidden states
                batch_hidden_state = batch_hidden_vectors.astype(np.float16)
                batch_text_fixed = np.array(generated_texts, dtype=TEXT_DTYPE)

            else:
                # Mode 2: Direct hidden state extraction (no text generation)
                # ~34x faster, returns (N, hidden_size) last-token hidden states
                batch_hidden_vectors, metrics = extractor.extract_hidden_states_no_cot(
                    batch_observations
                )
                batch_hidden_state = batch_hidden_vectors.astype(np.float16)
                batch_text_fixed = np.array(["" for _ in range(current_batch_size)], dtype=TEXT_DTYPE)

            # 4. SAVE PROGRESS TO DISK
            hidden_states_memmap[current_batch_indices, :] = batch_hidden_state
            text_outputs_memmap[current_batch_indices] = batch_text_fixed

            # Flush mmap files
            hidden_states_memmap.flush()
            text_outputs_memmap.flush()
            
            # Update the progress file
            write_progress(progress_path, current_batch_idx)

            logger.info(f"  ... completed batch {current_batch_idx} / {num_samples // BATCH_SIZE}")
            wandb.log({"progress_batches": current_batch_idx, "total_batches": num_samples // BATCH_SIZE})

        end_time = time.time()
        logger.info(f"Finished inference in {end_time - start_time:.2f}s for {file_path}")

        # 5. FINAL PACKAGING AND CLEANUP
        del hidden_states_memmap
        del text_outputs_memmap
        
        logger.info("Loading temporary .npy files for final save...")
        hidden_states_numpy = np.memmap(
            temp_hidden_path, dtype=np.float16, mode='r',
            shape=(num_samples, HIDDEN_SIZE)
        )
        all_outputs_numpy = np.memmap(
            temp_text_path, dtype=TEXT_DTYPE, mode='r', shape=(num_samples,)
        ).astype(object)  # Convert back to object

        save_data = {
            "obs": data["obs"], "next_obs": data["next_obs"],
            "action": data["action"], "reward": data["reward"],
            "done": data["done"], "log_prob": data["log_prob"],
            "hidden_state": hidden_states_numpy,
            "text_generated": all_outputs_numpy
        }

        result_path = os.path.join(RESULTS_DIR, job_basename)

        logger.info(f"Saving final augmented data to: {result_path}")
        np.savez_compressed(result_path, **save_data)
        logger.info(f"Job {file_path} completed and saved.")

        # Clean up ALL temporary files on success
        os.remove(temp_hidden_path)
        os.remove(temp_text_path)
        os.remove(progress_path)
        
        wandb.finish()

    except Exception as e:
        # 6. HANDLE ERRORS
        # We DO NOT re-queue. The Janitor will handle it.
        # We DO NOT delete the temp files. The next worker needs them.
        logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
        if wandb.run:
            wandb.finish(exit_code=1)

logger.info("Worker finished.")
