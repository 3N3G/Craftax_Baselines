"""
LLM Worker for Craftax Data Labelling

This script processes trajectory data through a text-only LLM (Qwen3-4B-Thinking-2507)
to extract hidden state representations from game observations.

Based on preempt_safe_worker.py but adapted for:
- Text-only LLM instead of VLM
- render_craftax_text() for observations
- Prompt format from vlm_play.py
"""

import redis
import numpy as np
import os
import time
import logging
import socket
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
from obs_to_text import obs_to_text  # Decode symbolic observations to text

# --- Constants ---
QUEUE_NAME = "craftax_llm_job_queue"  # Separate queue name to avoid conflicts
RESULTS_DIR = "/data/group_data/rl/geney/craftax_llm_labelled_results/"
LOGS_DIR = "/data/group_data/rl/geney/craftax_llm_job_logs/"
TEMP_NPY_DIR = os.path.join(RESULTS_DIR, "temp_npy")
PROGRESS_DIR = os.path.join(RESULTS_DIR, "progress")

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
BATCH_SIZE = 16  # 32 OOMed, 16 should be safe with ~50% headroom
TOKENS_GENERATED = 256  # Token budget for thinking + answer

# --- mmap Constants ---
MAX_TEXT_LEN = 2048
TEXT_DTYPE = f'<U{MAX_TEXT_LEN}'

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

# --- Model Initialization ---
logger.info(f"Initializing {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
logger.info(f"{MODEL_ID} initialized.")

HIDDEN_SIZE = model.config.hidden_size
# For Qwen models, we may need to downsample the sequence
# This will be determined based on actual generation length
logger.info(f"Model hidden size: {HIDDEN_SIZE}")

# --- Background tiles to filter out (from llm_play_harnessed.py) ---
BACKGROUND_TILES = {
    "grass", "sand", "gravel", 
    "fire grass", "ice grass", "fire_grass", "ice_grass"
}

def filter_text_obs(text_obs: str) -> str:
    """
    Filter out background tiles from the text observation to reduce token count
    and help the model focus on interesting/interactive tiles.
    
    Handles the compact Map: format from obs_to_text:
    Map: -5,-4:grass, -4,-4:tree, ...
    
    Args:
        text_obs: The full text observation from obs_to_text()
    
    Returns:
        Filtered observation with only interesting tiles shown
    """
    lines = text_obs.split('\n')
    filtered_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Detect Map line (compact format from obs_to_text)
        if stripped.startswith('Map:'):
            # Parse the compact map format
            map_content = stripped[4:].strip()  # Remove "Map:" prefix
            tiles = [t.strip() for t in map_content.split(',') if ':' in t]
            
            interesting_tiles = []
            for tile in tiles:
                # Handle compound tiles like "-5,-4:grass" 
                # Find the last colon which separates coord from tile type
                parts = tile.rsplit(':', 1)
                if len(parts) == 2:
                    coord = parts[0].strip()
                    tile_type = parts[1].strip().lower()
                    
                    # Check if tile is interesting (not pure background)
                    is_background = tile_type in BACKGROUND_TILES
                    has_entity = ' on ' in tile_type  # e.g., "Cow on grass"
                    
                    if not is_background or has_entity:
                        interesting_tiles.append(f"{coord}:{parts[1].strip()}")
            
            if interesting_tiles:
                filtered_lines.append(f"Map (interesting tiles only): {', '.join(interesting_tiles)}")
            else:
                filtered_lines.append("Map: [No interesting tiles in view - all background]")
            continue
        
        # Keep all non-map lines
        if stripped:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


# --- Prompt Configuration (from llm_play_harnessed.py) ---
SYSTEM_PROMPT = """You are playing Craftax.

Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions and can interact. Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest. This interaction will only happen if the block/creature/chest is directly in front of the player, one step in the direction the player is facing. 
The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. Mana is used for casting spells or enchanting items and will naturally recover. Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. If the player's health falls beneath 0 they will die and the game will restart.

The coordinate system is (Row, Column). Everything is relative to your current position, and the map will show all interesting tiles (so tiles with something besides grass or other background tiles that you will always be able to walk on) within 5 columns and 4 rows of your current position.
- Negative Row is UP. Positive Row is DOWN.
- Negative Column is LEFT. Positive Column is RIGHT.
- (0, 0) is your current position.
- Example: (-1, 0) is one step UP. (0, 1) is one step RIGHT.

To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). There are 9 levels in total.

### GAMEPLAY ALGORITHM
The player should focus on three main aspects: staying alive, collecting resources or tools, and progressing.

**1. Check Intrinsics First**
Look at your health, food, drink, and energy. Without leveling up any stats, the max for each stat is 9.
- If your health is low or medium, make sure to fill up your food, drink, and energy in order to recover health.
- If any of your other intrinsics are medium, either recover it immediately or make sure you will have access to that resource once it becomes low.
- For food and drink: consume animals (cows, snails, bats, etc.) or drink water.
- For energy: you need to sleep. Make sure you are protected (e.g. closed off by stone walls) before going to sleep, otherwise enemies will attack and kill you.

**2. Collect Resources and Tools**
If your intrinsics are fine, collect resources and tools:
- Mine trees with your hand. Note that extra wood is helpful for crafting torches later.
- Once you have at least 2 wood, craft a crafting table. Then if you have wood, craft a wood pickaxe and sword.
- Mine stone and then craft a stone pickaxe and sword. Also mine coal whenever you see it, as it is helpful for crafting iron tools and torches. Note that extra stone is helpful for blocking off enemies to rest and recover.
- If you see iron, mine it, and if you have at least one iron, one coal, and one wood, you can craft an iron sword or pickaxe. This step is not as urgent since sometimes there may not be enough iron for all of these.

**3. Progress**
This means finding the ladder, which means looking for it and (on all floors after the overworld) killing 8 troops. If you see the ladder, and it is open, enter it. Otherwise keep exploring and staying alive.

Actions available: 
0:NOOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:DO (interact/mine/attack), 6:SLEEP, 7:PLACE_STONE,
8:PLACE_TABLE, 9:PLACE_FURNACE, 10:PLACE_PLANT, 11:MAKE_WOOD_PICKAXE, 12:MAKE_STONE_PICKAXE,
13:MAKE_IRON_PICKAXE, 14:MAKE_WOOD_SWORD, 15:MAKE_STONE_SWORD, 16:MAKE_IRON_SWORD, 17:REST,
18:DESCEND, 19:ASCEND, 20:MAKE_DIAMOND_PICKAXE, 21:MAKE_DIAMOND_SWORD, 22:MAKE_IRON_ARMOUR,
23:MAKE_DIAMOND_ARMOUR, 24:SHOOT_ARROW, 25:MAKE_ARROW, 26:CAST_FIREBALL, 27:CAST_ICEBALL,
28:PLACE_TORCH, 29-34:DRINK_POTION_(RED/GREEN/BLUE/PINK/CYAN/YELLOW), 35:READ_BOOK,
36:ENCHANT_SWORD, 37:ENCHANT_ARMOUR, 38:MAKE_TORCH, 39-41:LEVEL_UP_(DEX/STR/INT), 42:ENCHANT_BOW

### COORDINATE & INTERACTION RULES
1. Matrix Coordinates: The map uses (Row, Column) notation. All relevant tiles are shown (for brevity, grass and other background tiles are filtered out). 
   - ROW (1st number): Vertical. Negative is UP, Positive is DOWN.
   - COL (2nd number): Horizontal. Negative is LEFT, Positive is RIGHT.
   - Example: From (0,0), moving UP goes to (-1, 0). Moving DOWN goes to (1, 0).

2. Interaction Targets: The 'DO' action only affects the block you are FACING. You will face the direction you most recently moved in.
   - If facing UP: You act on (-1, 0)
   - If facing DOWN: You act on (1, 0)
   - If facing LEFT: You act on (0, -1)
   - If facing RIGHT: You act on (0, 1)

Note: Do NOT reference coordinates from the examples - those are just to show you how to strategically plan your moves. Only use the CURRENT GAME STATE.
"""

few_shot_examples = """
--- EXAMPLE 1 ---
Game State:
Map (interesting tiles only): 1, 0:crafting_table, 2, -3:tree, 4, 0:stone, 5, 0:stone
Inventory:
Wood: 3
...
Health: 9.0
Food: 9
Drink: 9
Energy: 9
Mana: 9
Direction: right
...
<think> I have full health. I am not hungry. I am not thirsty. I am not tired. I have 3 Wood and I need to craft tools. I see a crafting table already here! I see a 'crafting_table' at (1, 0).
Let me calculate where it is. Crafting table (1, 0) - Me (0,0) = (+1 Row, 0 Col). +1 Row is DOWN. My Direction is 'right'. I need to face 'down' to interact. I must move. Since it is adjacent, I can just interact if I face it. I will move DOWN to face the table.
</think>
**Action:** 4 (DOWN)
   

--- EXAMPLE 2 ---
Game State:
Map (interesting tiles only): 0, -3:tree, 1, 3:stone
Inventory:
Wood: 0
...
Direction: right
...
<think>
I have full health. I am not hungry. I am not thirsty. I have 0 Wood. I need to find a tree to gather wood. Scanning the map...
1. Scan Map: Found 'tree' at (0, -3).
2. Calculate Position:
   - Target: (0, -3)
   - Me: (0, 0)
   - Delta: (0 Row, -3 Col).
   - Analysis: Same row, 3 steps LEFT.
3. Plan: The tree is to my WEST (Left). I am facing EAST (Right). I need to walk over there to chop it.
   - First step: Move LEFT.
</think>
**Action:** 1 (LEFT)

--- EXAMPLE 3 ---
Game State:
Map (interesting tiles only): 0, -2:water, 3, -4:tree, -3, -3:tree, 4, -3:Skeleton on grass
Inventory:
...
Health: 4.0
Food: 5
Drink: 4
...
Direction: down
...
<think>
I am thirsty (Drink: 4). I have low health (4/10). I need to recover health, so I need to eat and drink water. There is water nearby. The skeleton is far away, so I don't need to worry about it.
1. Scan Map: Found 'water' at (0, -2).
2. Calculate Position:
   - Target (0, -2). Me (0, 0).
   - Delta: (0, -2).
   - Analysis: 2 steps LEFT.
3. Plan: This is a simple situation with a clear single objective: reach the water. No threats nearby. I will move LEFT.
</think>
**Action:** 1 (LEFT)

--- EXAMPLE 4 ---
Game State:
Map (interesting tiles only): 0, 2:Arrow , 0, 3:Skeleton, 1, 1:Cow
Inventory:
Wood: 0
...
Health: 3.0
...
Direction: right
...
<think>
I am in critical condition (Health: 3.0). I am hungry. I see a Skeleton to my East shooting at me! There is an Arrow incoming at (0, 2).
1. Analyze Threats:
   - 'Arrow' at (0, 2). It is on my row (Row 0). It will hit me if I stay or move East/West in this row.
   - 'Skeleton' at (0, 3).
2. Analyze Resources:
   - 'Cow' at (1, 1). This is food/healing.
3. Calculate Position:
   - Me: (0, 0).
   - Arrow: (0, 2).
   - Cow: (1, 1).
4. Strategic Decision:
   - I MUST dodge the arrow immediately. I cannot fight (health too low).
   - Moving DOWN (to 1, 0) takes me to Row 1, safely out of the arrow's path (Row 0).
   - Moving DOWN also places me at (1, 0), which is adjacent to the Cow at (1, 1).
   - This achieves two goals: Dodge the threat AND approach healing.
5. Plan: Move DOWN.
</think>
**Action:** 4 (DOWN)

--- EXAMPLE 5 ---
Game State:
Map (interesting tiles only): 1, 0:Orc Soldier on torch on path, 4, 0:Snail on path
Inventory:
Wood: 17
Iron: 4
Stone: 61
Iron Sword with No enchantment
Iron Helmet with No enchantment
...
Health: 6.3
Food: 3
...
Direction: down
...
<think>
I have decent health (6.3). I am hungry. I am not thirsty. I am not tired. There is an Orc Soldier one tile below me. I have Iron gear (Sword, Armor). I am confident I can win.
1. Scan Map: 'Orc Soldier' at (1, 0).
2. Calculate Position:
   - Target (1, 0) - Me (0, 0) = (+1 Row, 0 Col).
   - Analysis: The enemy is 1 step DOWN.
3. Orientation Check: My Direction is 'down'.
   - Result: I am facing the enemy.
4. Strategy: Fight. I have armor and a sword. I need to kill 8 enemies to open the ladder. I probably will not die because I have armor and a sword. I will attack to clear the path and potentially get loot.
Action: DO (Attack).
</think>
**Action:** 5 (DO)

--- END OF EXAMPLES ---
==================================================
>>> LIVE ENVIRONMENT STREAM STARTS HERE <<<
>>> IGNORE ALL COORDINATES FROM EXAMPLES ABOVE <<<
==================================================
"""

def create_prompt(text_obs):
    """Create prompt from text observation using llm_play_harnessed.py format.
    
    Args:
        text_obs: The filtered text observation (should already have filter_text_obs applied)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Below are examples of good gameplay decisions. These are EXAMPLES ONLY, not your actual game history:\n{few_shot_examples}\nYOUR CURRENT GAME STATE (use ONLY this map for coordinates):\n{text_obs}\n\nYou are at (0,0). Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>)."},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        
        # Determine hidden state dimensions from first sample
        # We'll save the full generated hidden states (no fixed downsampling upfront)
        # The actual sequence length will vary, so we use max expected
        MAX_SEQ_LEN = TOKENS_GENERATED  # Maximum possible generated tokens
        
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch} (sample index {start_index})")
            # Open existing files  in read/write ('r+') mode
            hidden_states_memmap = np.memmap(
                temp_hidden_path, dtype=np.float16, mode='r+', 
                shape=(num_samples, MAX_SEQ_LEN, HIDDEN_SIZE)
            )
            text_outputs_memmap = np.memmap(
                temp_text_path, dtype=TEXT_DTYPE, mode='r+', shape=(num_samples,)
            )
        else:
            logger.info("Starting new job, creating temp files.")
            # Create new files in write ('w+') mode
            hidden_states_memmap = np.memmap(
                temp_hidden_path, dtype=np.float16, mode='w+', 
                shape=(num_samples, MAX_SEQ_LEN, HIDDEN_SIZE)
            )
            text_outputs_memmap = np.memmap(
                temp_text_path, dtype=TEXT_DTYPE, mode='w+', shape=(num_samples,)
            )
            # Write initial progress file
            write_progress(progress_path, -1)

        # 3. RUN INFERENCE (THE LONG PART)
        logger.info(f"Beginning inference from index {start_index}...")
        start_time = time.time()

        for i in range(start_index, num_samples, BATCH_SIZE):
            current_batch_idx = i // BATCH_SIZE
            batch_prompts = []
            current_batch_indices = range(i, min(i + BATCH_SIZE, num_samples))
            current_batch_size = len(current_batch_indices)

            # Get text observations by decoding symbolic obs
            for idx in current_batch_indices:
                # Use pre-saved text_obs if available, otherwise decode from obs
                if "text_obs" in data and data["text_obs"][idx]:
                    raw_text_obs = str(data["text_obs"][idx])
                else:
                    # Decode symbolic observation to text
                    raw_text_obs = obs_to_text(data["obs"][idx])
                
                # Filter to show only interesting tiles (remove background)
                filtered_text_obs = filter_text_obs(raw_text_obs)
                prompt = create_prompt(filtered_text_obs)
                batch_prompts.append(prompt)

            # Tokenize all prompts in batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(model.device)

            prompt_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=TOKENS_GENERATED,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Extract hidden states from last layer
            # outputs.hidden_states is a tuple of tuples: (step, layer, batch, seq, hidden)
            last_layer_states_list = [s[-1] for s in outputs.hidden_states]
            generated_hidden_states = torch.cat(last_layer_states_list, dim=1)
            
            # Handle variable sequence lengths by padding/truncating to MAX_SEQ_LEN
            seq_len = generated_hidden_states.shape[1]
            if seq_len > MAX_SEQ_LEN:
                generated_hidden_states = generated_hidden_states[:, :MAX_SEQ_LEN, :]
            elif seq_len < MAX_SEQ_LEN:
                padding = torch.zeros(
                    (current_batch_size, MAX_SEQ_LEN - seq_len, HIDDEN_SIZE),
                    device=generated_hidden_states.device,
                    dtype=generated_hidden_states.dtype
                )
                generated_hidden_states = torch.cat([generated_hidden_states, padding], dim=1)
            
            batch_hidden_state = generated_hidden_states.cpu().to(torch.float16).numpy()

            # --- Text Output Processing ---
            generated_token_ids = outputs.sequences[:, prompt_len:]
            generated_text_list = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            batch_text_fixed = np.array(generated_text_list, dtype=TEXT_DTYPE)

            # 4. SAVE PROGRESS TO DISK
            hidden_states_memmap[current_batch_indices, :, :] = batch_hidden_state
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
        hidden_states_numpy = np.load(temp_hidden_path)
        all_outputs_numpy = np.load(temp_text_path).astype(object)  # Convert back to object

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
