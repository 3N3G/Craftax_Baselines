#!/usr/bin/env python3
"""
Fast Hidden State Extraction via Prefill

This script extracts hidden states from already-generated text by running a single
forward pass (prefill) instead of regenerating. This is 10-30x faster than generation.

The hidden states are mathematically identical to what autoregressive generation produces:
- During generation step n: model sees tokens 0..n, outputs hidden[n]
- During prefill with tokens 0..N: we get hidden[0..N] for all positions
- Position n in prefill == hidden state at generation step n

IMPORTANT: This produces EXACTLY the same hidden states as llm_worker.py and 
online_rl_hidden.py. The output shape is (batch, 256, hidden_size) - matching the
256 generation tokens stored by llm_worker.

Usage:
    python extract_hidden_states.py --input /path/to/vllm_output.npz --output /path/to/with_hidden.npz
    
For batch processing (Redis queue mode):
    python extract_hidden_states.py --queue-mode
"""

import argparse
import os
import sys
import time
import logging
import socket
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Constants ---
QUEUE_NAME = "craftax_hidden_extraction_queue"
INPUT_DIR = "/data/group_data/rl/geney/craftax_llm_labelled_results/"  # vLLM outputs
OUTPUT_DIR = "/data/group_data/rl/geney/new_craftax_llm_labelled_results/"  # With hidden states
LOGS_DIR = "/data/group_data/rl/geney/craftax_hidden_extraction_logs/"

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# MUST MATCH llm_worker.py / online_rl_hidden.py exactly!
TOKENS_GENERATED = 256  # Same as other scripts
MAX_SEQ_LEN = 2560  # Prompt (~1500-2000) + Generated (256)

# Memory: Batch 16 works - tested previously
BATCH_SIZE = 16

# For output format compatibility
MAX_TEXT_LEN = 2048
TEXT_DTYPE = f'<U{MAX_TEXT_LEN}'

# --- Background tiles to filter (matching other scripts) ---
BACKGROUND_TILES = {
    "grass", "sand", "gravel", 
    "fire grass", "ice grass", "fire_grass", "ice_grass"
}

# --- Logging Setup ---
pid = os.getpid()
hostname = socket.gethostname()


def setup_logging(mode: str = "file"):
    """Setup logging to file and/or stdout."""
    logger = logging.getLogger(f"hidden_extractor_{pid}")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Always log to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # Optionally log to file
    if mode == "file":
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_filename = os.path.join(LOGS_DIR, f"extractor_{hostname}_{pid}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# --- System Prompt (must match what was used for generation) ---
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

FEW_SHOT_EXAMPLES = """
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

--- END OF EXAMPLES ---
==================================================
>>> LIVE ENVIRONMENT STREAM STARTS HERE <<<
>>> IGNORE ALL COORDINATES FROM EXAMPLES ABOVE <<<
==================================================
"""


def filter_text_obs(text_obs: str) -> str:
    """Filter out background tiles from the text observation."""
    lines = text_obs.split('\n')
    filtered_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('Map:'):
            map_content = stripped[4:].strip()
            tiles = [t.strip() for t in map_content.split(',') if ':' in t]
            
            interesting_tiles = []
            for tile in tiles:
                parts = tile.rsplit(':', 1)
                if len(parts) == 2:
                    coord = parts[0].strip()
                    tile_type = parts[1].strip().lower()
                    
                    is_background = tile_type in BACKGROUND_TILES
                    has_entity = ' on ' in tile_type
                    
                    if not is_background or has_entity:
                        interesting_tiles.append(f"{coord}:{parts[1].strip()}")
            
            if interesting_tiles:
                filtered_lines.append(f"Map (interesting tiles only): {', '.join(interesting_tiles)}")
            else:
                filtered_lines.append("Map: [No interesting tiles in view - all background]")
            continue
        
        if stripped:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def create_prompt(text_obs: str) -> str:
    """Create prompt from text observation."""
    return f"Below are examples of good gameplay decisions. These are EXAMPLES ONLY, not your actual game history:\n{FEW_SHOT_EXAMPLES}\nYOUR CURRENT GAME STATE (use ONLY this map for coordinates):\n{text_obs}\n\nYou are at (0,0). Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>)."


def process_npz_file(
    input_path: str,
    output_path: str,
    model,
    tokenizer,
    logger,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """
    Extract hidden states from already-generated text using prefill.
    
    This is mathematically equivalent to generation because:
    - The hidden state at position n during autoregressive generation equals
    - The hidden state at position n during a forward pass over the same sequence
    
    IMPORTANT: We extract EXACTLY 256 hidden states per sample (TOKENS_GENERATED),
    matching llm_worker.py. The output shape is (num_samples, 256, hidden_size).
    
    Args:
        input_path: Path to vLLM output NPZ with 'text_generated' key
        output_path: Path to save NPZ with 'hidden_state' key added
        model: Loaded HuggingFace model
        tokenizer: Loaded tokenizer
        logger: Logger instance
        batch_size: Batch size for processing
    """
    from obs_to_text import obs_to_text
    
    logger.info(f"Processing {input_path}")
    
    # Load input data
    data = np.load(input_path, allow_pickle=True)
    num_samples = len(data["obs"])
    
    if "text_generated" not in data.files:
        raise ValueError(f"Input file missing 'text_generated' key: {input_path}")
    
    text_generated = data["text_generated"]
    logger.info(f"Loaded {num_samples} samples with generated text")
    
    # Get hidden size from model
    hidden_size = model.config.hidden_size
    
    # Pre-allocate hidden states array
    # Shape: (num_samples, TOKENS_GENERATED, hidden_size) = (num_samples, 256, hidden_size)
    # This matches what online_rl_hidden.py gets BEFORE mean pooling
    all_hidden_states = np.zeros((num_samples, TOKENS_GENERATED, hidden_size), dtype=np.float16)
    
    start_time = time.time()
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_size_actual = batch_end - batch_start
        
        # Build full sequences: prompt + generated text
        full_texts = []
        prompt_lengths = []
        
        for i in range(batch_start, batch_end):
            # Reconstruct prompt from observation
            if "text_obs" in data.files and data["text_obs"][i]:
                raw_text_obs = str(data["text_obs"][i])
            else:
                raw_text_obs = obs_to_text(data["obs"][i])
            
            filtered_text_obs = filter_text_obs(raw_text_obs)
            
            # Build chat messages (prompt only)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": create_prompt(filtered_text_obs)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Get the generated text
            generated = str(text_generated[i]) if text_generated[i] else ""
            
            # Full sequence = prompt + generated
            full_text = prompt + generated
            full_texts.append(full_text)
            
            # Track prompt length for later extraction
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_lengths.append(len(prompt_tokens))
        
        # Tokenize all sequences with padding
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(model.device)
        
        # Forward pass - ONLY get last_hidden_state (not all 36 layers!)
        # This is much more memory efficient than output_hidden_states=True
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=False,  # Don't store all layers!
                return_dict=True,
            )
        
        # Get last hidden state directly from model output
        # Shape: (batch, seq_len, hidden_size)
        # For CausalLM models, this is NOT directly available - we need to get it differently
        # Actually, Qwen3 CausalLM outputs don't have last_hidden_state by default
        # We need to get it from the base model
        
        # Get hidden states from the base model's output
        # The CausalLM wrapper adds the lm_head on top
        # We can access the last hidden state from the underlying model
        if hasattr(model, 'model'):
            # Get the base model
            with torch.no_grad():
                base_outputs = model.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
            last_hidden = base_outputs.last_hidden_state
        else:
            # Fallback: use output_hidden_states for just the last layer
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden = outputs.hidden_states[-1]
            del outputs.hidden_states[:-1]  # Free memory for other layers
        
        # For each sample, extract ALL hidden states from the generated portion
        # This matches online_rl_hidden.py (which then mean-pools for training)
        for i, prompt_len in enumerate(prompt_lengths):
            sample_idx = batch_start + i
            seq_len = inputs.attention_mask[i].sum().item()  # Actual sequence length
            
            # Generated tokens start after prompt
            gen_start = prompt_len
            gen_end = min(seq_len, prompt_len + TOKENS_GENERATED)  # Cap at 256 tokens
            gen_len = gen_end - gen_start
            
            if gen_len <= 0:
                logger.warning(f"Sample {sample_idx}: No generated tokens found")
                continue
            
            # Extract ALL hidden states for generated tokens (no subsampling)
            hidden_slice = last_hidden[i, gen_start:gen_end, :].cpu().to(torch.float16).numpy()
            all_hidden_states[sample_idx, :gen_len, :] = hidden_slice
            # Remaining positions stay zero if gen_len < TOKENS_GENERATED
        
        # Memory cleanup
        del last_hidden
        if 'base_outputs' in dir():
            del base_outputs
        torch.cuda.empty_cache()
        
        if (batch_start // batch_size + 1) % 5 == 0:
            elapsed = time.time() - start_time
            samples_done = batch_end
            rate = samples_done / elapsed
            eta = (num_samples - samples_done) / rate if rate > 0 else 0
            logger.info(f"  Batch {batch_start // batch_size + 1}: {samples_done}/{num_samples} "
                       f"({rate:.1f} samples/sec, ETA: {eta:.0f}s)")
    
    elapsed = time.time() - start_time
    logger.info(f"Extraction complete in {elapsed:.2f}s ({num_samples / elapsed:.2f} samples/sec)")
    
    # Save with hidden states
    save_data = {
        "obs": data["obs"],
        "next_obs": data["next_obs"],
        "action": data["action"],
        "reward": data["reward"],
        "done": data["done"],
        "log_prob": data["log_prob"],
        "text_generated": data["text_generated"],
        "hidden_state": all_hidden_states,  # Shape: (num_samples, 256, hidden_size)
    }
    
    logger.info(f"Saving to {output_path}")
    logger.info(f"Hidden states shape: {all_hidden_states.shape}")  # Should be (N, 256, 2560)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    np.savez_compressed(output_path, **save_data)
    
    return {
        "input_path": input_path,
        "output_path": output_path,
        "num_samples": num_samples,
        "hidden_states_shape": list(all_hidden_states.shape),
        "time_s": elapsed,
        "samples_per_sec": num_samples / elapsed,
    }


def queue_mode(args, logger, model, tokenizer):
    """Process files from Redis queue."""
    import redis
    
    REDIS_HOST_FILE = "/data/group_data/rl/geney/redis_host.txt"
    try:
        with open(REDIS_HOST_FILE, 'r') as f:
            REDIS_HOST = f.read().strip()
        logger.info(f"Read Redis host from file: {REDIS_HOST}")
    except FileNotFoundError:
        REDIS_HOST = "localhost"
        logger.warning(f"Redis host file not found, using: {REDIS_HOST}")
    
    r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    r.ping()
    logger.info(f"Connected to Redis at {REDIS_HOST}:6379")
    
    while True:
        file_path = r.rpop(args.queue_name)
        if file_path is None:
            logger.info("No more jobs! Exiting.")
            break
        
        logger.info(f"Processing job: {file_path}")
        job_basename = os.path.basename(file_path)
        output_path = os.path.join(args.output_dir, job_basename)
        
        try:
            stats = process_npz_file(
                input_path=file_path,
                output_path=output_path,
                model=model,
                tokenizer=tokenizer,
                logger=logger,
                batch_size=args.batch_size,
            )
            logger.info(f"Completed job: {file_path}")
            logger.info(f"Stats: {json.dumps(stats, indent=2)}")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
    
    logger.info("Worker finished.")


def main():
    parser = argparse.ArgumentParser(description="Fast hidden state extraction via prefill")
    parser.add_argument("--input", type=str, help="Input NPZ file (vLLM output with text_generated)")
    parser.add_argument("--output", type=str, help="Output NPZ file (with hidden_state added)")
    parser.add_argument("--queue-mode", action="store_true", help="Process files from Redis queue")
    parser.add_argument("--queue-name", type=str, default=QUEUE_NAME, help="Redis queue name")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR, help="Input directory for queue mode")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    
    logger = setup_logging("file" if args.queue_mode else "stdout")
    
    # Load model with Flash Attention 2
    logger.info(f"Loading model {args.model} with Flash Attention 2...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # CRITICAL: Use Flash Attention!
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    logger.info(f"Model loaded. Hidden size: {model.config.hidden_size}")
    
    if args.queue_mode:
        queue_mode(args, logger, model, tokenizer)
    elif args.input and args.output:
        stats = process_npz_file(
            input_path=args.input,
            output_path=args.output,
            model=model,
            tokenizer=tokenizer,
            logger=logger,
            batch_size=args.batch_size,
        )
        logger.info(f"Done! Stats: {json.dumps(stats, indent=2)}")
    else:
        parser.print_help()
        print("\nError: Must specify --input and --output, or use --queue-mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
