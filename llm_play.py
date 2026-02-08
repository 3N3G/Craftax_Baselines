"""Simple llm agent that plays Craftax using text observations."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_text, render_craftax_pixels
from craftax.craftax.constants import Action, Achievement, BLOCK_PIXEL_SIZE_AGENT
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import wandb
import cv2
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Default model - can be overridden with --model flag
# Supported Qwen3 sizes: 0.6B, 1.7B, 4B, 8B, 14B, 32B
# Format: Qwen/Qwen3-{size}B or Qwen/Qwen3-{size}B-Thinking-2507
DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"

SYSTEM_PROMPT = """You are playing Craftax.

Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions and can interact. Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest. This interaction will only happen if the block/creature/chest is directly in front of the player, one step in the direction the player is facing. 
The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. Mana is used for casting spells or enchanting items and will naturally recover. Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. If the player's health falls beneath 0 they will die and the game will restart.

The coordinate system is (Row, Column).
- Negative Row is UP. Positive Row is DOWN.
- Negative Column is LEFT. Positive Column is RIGHT.
- (0, 0) is your current position.
- Example: (-1, 0) is one step UP. (0, 1) is one step RIGHT.

To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). There are 9 levels in total.

Here is a rough outline of an example progression of this game:
- Gather wood from nearby trees to build a crafting table and then wood tools
- Find and mine stones to make stone tools or place as a furnace or stone
- Use your tools to mine iron ore and coal to use for building iron tools and armor, and make sure to collect extra coal and wood to use as torches later in the game
- The ladder is always open on the overworld, but once you have iron tools and torches you should be ready to descend. Traverse the overworld looking for the ladder and descend.
- Continue killing creatures, then finding and descending down ladders

Make sure to stay healthy during this process, killing cows (or that level's equivalent peaceful animal), drinking water, and resting enough to keep the intrinsics above 0. 

Actions available: 
0:NOOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:DO (interact/mine/attack), 6:SLEEP, 7:PLACE_STONE,
8:PLACE_TABLE, 9:PLACE_FURNACE, 10:PLACE_PLANT, 11:MAKE_WOOD_PICKAXE, 12:MAKE_STONE_PICKAXE,
13:MAKE_IRON_PICKAXE, 14:MAKE_WOOD_SWORD, 15:MAKE_STONE_SWORD, 16:MAKE_IRON_SWORD, 17:REST,
18:DESCEND, 19:ASCEND, 20:MAKE_DIAMOND_PICKAXE, 21:MAKE_DIAMOND_SWORD, 22:MAKE_IRON_ARMOUR,
23:MAKE_DIAMOND_ARMOUR, 24:SHOOT_ARROW, 25:MAKE_ARROW, 26:CAST_FIREBALL, 27:CAST_ICEBALL,
28:PLACE_TORCH, 29-34:DRINK_POTION_(RED/GREEN/BLUE/PINK/CYAN/YELLOW), 35:READ_BOOK,
36:ENCHANT_SWORD, 37:ENCHANT_ARMOUR, 38:MAKE_TORCH, 39-41:LEVEL_UP_(DEX/STR/INT), 42:ENCHANT_BOW

### COORDINATE & INTERACTION RULES
1. Matrix Coordinates: The map uses (Row, Column) notation.
   - ROW (1st number): Vertical. Negative is UP, Positive is DOWN.
   - COL (2nd number): Horizontal. Negative is LEFT, Positive is RIGHT.
   - Example: From (0,0), moving UP goes to (-1, 0). Moving DOWN goes to (1, 0).

2. Interaction Targets: The 'DO' action only affects the block you are FACING.
   - If facing UP: You act on (Row-1, Col)
   - If facing DOWN: You act on (Row+1, Col)
   - If facing LEFT: You act on (Row, Col-1)
   - If facing RIGHT: You act on (Row, Col+1)

### IMPORTANT: GROUNDING REQUIREMENT
Before planning your move, you MUST explicitly check the CURRENT MAP observation for:
1. Your immediate neighbors: (0,1), (0,-1), (1,0), and (-1,0)
2. The specific objects you are targeting
Do NOT reference coordinates from examples above - only from the CURRENT GAME STATE.
"""
# should we include which achievements exist / which have been gotten in the prompt?


def load_model(model_id: str = None):
    """Load a Qwen model. Supports all Qwen3 sizes (0.6B, 1.7B, 4B, 8B, 14B, 32B)."""
    model_id = model_id or DEFAULT_MODEL
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"Model loaded! Hidden size: {model.config.hidden_size}")
    return model, processor

def get_action(model, processor, text_obs: str) -> tuple[int, dict]:
    """
    Get action from model, handling timeouts and logging full context.
    
    Returns:
        action (int): The selected action (0-42).
        log_data (dict): Contains 'full_prompt', 'full_response', 'thinking', 'content', etc.
    """
    SPLIT_TOKEN_ID = 151668  # </think>
    
    # 1. Prepare Inputs & Full Prompt
    # Few-shot examples showing high-quality gameplay patterns
    # These are formatted WITHOUT chat markers to avoid confusing the model into thinking they're actual history
    few_shot_examples = """
--- EXAMPLE 1: Placing a crafting table ---
Game State:
Map:
-5, -4: grass
-4, -4: grass
-3, -4: grass
-2, -4: sand
-1, -4: water
0, -4: sand
1, -4: tree
2, -4: grass
3, -4: grass
4, -4: grass
5, -4: grass
-5, -3: grass
-4, -3: grass
-3, -3: grass
-2, -3: sand
-1, -3: water
0, -3: sand
1, -3: grass
2, -3: tree
3, -3: grass
4, -3: grass
5, -3: grass
-5, -2: grass
-4, -2: grass
-3, -2: grass
-2, -2: grass
-1, -2: water
0, -2: sand
1, -2: grass
2, -2: grass
3, -2: grass
4, -2: grass
5, -2: grass
-5, -1: grass
-4, -1: grass
-3, -1: grass
-2, -1: sand
-1, -1: sand
0, -1: sand
1, -1: grass
2, -1: grass
3, -1: grass
4, -1: grass
5, -1: grass
-5, 0: grass
-4, 0: grass
-3, 0: grass
-2, 0: sand
-1, 0: sand
0, 0: sand
1, 0: grass
2, 0: grass
3, 0: grass
4, 0: stone
5, 0: stone
-5, 1: grass
-4, 1: sand
-3, 1: sand
-2, 1: water
-1, 1: sand
0, 1: grass
1, 1: grass
2, 1: tree
3, 1: stone
4, 1: stone
5, 1: path
-5, 2: water
-4, 2: water
-3, 2: path
-2, 2: stone
-1, 2: sand
0, 2: grass
1, 2: grass
2, 2: stone
3, 2: stone
4, 2: stone
5, 2: path
-5, 3: coal
-4, 3: stone
-3, 3: path
-2, 3: water
-1, 3: sand
0, 3: grass
1, 3: grass
2, 3: stone
3, 3: stone
4, 3: iron
5, 3: path
-5, 4: stone
-4, 4: stone
-3, 4: water
-2, 4: water
-1, 4: sand
0, 4: grass
1, 4: grass
2, 4: grass
3, 4: stone
4, 4: stone
5, 4: path

Inventory:
Wood: 3
Stone: 0
Coal: 0
Iron: 0
Diamond: 0
Sapphire: 0
Ruby: 0
Sapling: 0
Torch: 0
Arrow: 0
Book: 0
Red potion: 0
Green potion: 0
Blue potion: 0
Pink potion: 0
Cyan potion: 0
Yellow potion: 0
Health: 9.0
Food: 9
Drink: 9
Energy: 9
Mana: 9
XP: 0
Dexterity: 1
Strength: 1
Intelligence: 1
Direction: right
Light: 0.8945
Is Sleeping: False
Is Resting: False
Learned Fireball: False
Learned Iceball: False
Floor: 0
Ladder Open: True
Is Boss Vulnerable: False
<think>
I have 3 Wood and I need to craft tools. I see a crafting table already here!
1. Scan Map: Found 'crafting_table' at (1, 0).
2. Calculate Position:
   - Target (1, 0) - Me (0,0) = (+1 Row, 0 Col).
   - +1 Row is DOWN.
3. Orientation Check: My Direction is 'right'. I need to face 'down' to interact.
4. Plan: I must turn or move. Since it is adjacent, I can just interact if I face it. Wait, I am not facing it. I should move towards it or turn. Actually, if I move DOWN I will stand on top of it? No, it's a solid block probably.
   - Better option: I will make my own table? No, I see one.
   - Let's correct: If I want to place a table, I check if one exists. One exists.
   - I will Interact. But I am facing Right.
   - Let's assume I want to PLACE a table in a clear spot.
   - Narrative: 'I have wood... I do NOT see a table (simulated)...'
   - Okay, let's stick to the user's Example 1: Placing a new table.
   
   RETRYING REASONING FOR PREVIOUS USER EXAMPLE:
   "I have 3 Wood. I am healthy. I see no crafting table in my immediate vicinity (the one at 1,0 is far? No, 1,0 is close).
   Actually, let's look at the Map provided in the code. It has '1, -4:tree'. No table in the map provided in my previous turn. The user removed my table from the map in their thought?
   Okay, let's stick to the MAP provided in the text block below."

   "I have 3 Wood. I need to make a pickaxe. I do not see a crafting table nearby.
   1. Scan: No 'crafting_table' found.
   2. Inventory: 3 Wood (Sufficient).
   3. Action: I will place a table in front of me to start crafting.
</think>
**Action:** 8 (PLACE_TABLE)

--- EXAMPLE 2: Moving to gather wood ---
Game State:
Map:
-5, -4: sand
-4, -4: grass
-3, -4: grass
-2, -4: grass
-1, -4: grass
0, -4: grass
1, -4: grass
2, -4: grass
3, -4: grass
4, -4: grass
5, -4: grass
-5, -3: sand
-4, -3: sand
-3, -3: grass
-2, -3: grass
-1, -3: grass
0, -3: tree
1, -3: grass
2, -3: grass
3, -3: grass
4, -3: grass
5, -3: grass
-5, -2: sand
-4, -2: sand
-3, -2: grass
-2, -2: grass
-1, -2: grass
0, -2: grass
1, -2: grass
2, -2: grass
3, -2: grass
4, -2: grass
5, -2: grass
-5, -1: water
-4, -1: sand
-3, -1: grass
-2, -1: grass
-1, -1: grass
0, -1: grass
1, -1: grass
2, -1: grass
3, -1: grass
4, -1: grass
5, -1: grass
-5, 0: water
-4, 0: sand
-3, 0: grass
-2, 0: grass
-1, 0: grass
0, 0: grass
1, 0: grass
2, 0: grass
3, 0: grass
4, 0: grass
5, 0: grass
-5, 1: water
-4, 1: sand
-3, 1: grass
-2, 1: grass
-1, 1: grass
0, 1: grass
1, 1: grass
2, 1: grass
3, 1: grass
4, 1: grass
5, 1: grass
-5, 2: water
-4, 2: sand
-3, 2: grass
-2, 2: grass
-1, 2: grass
0, 2: grass
1, 2: grass
2, 2: grass
3, 2: grass
4, 2: grass
5, 2: path
-5, 3: sand
-4, 3: sand
-3, 3: grass
-2, 3: grass
-1, 3: grass
0, 3: grass
1, 3: grass
2, 3: tree
3, 3: stone
4, 3: stone
5, 3: path
-5, 4: sand
-4, 4: sand
-3, 4: crafting_table
-2, 4: grass
-1, 4: grass
0, 4: stone
1, 4: stone
2, 4: coal
3, 4: stone
4, 4: stone
5, 4: path

Inventory:
Wood: 0
Stone: 0
Coal: 0
Iron: 0
Diamond: 0
Sapphire: 0
Ruby: 0
Sapling: 0
Torch: 0
Arrow: 0
Book: 0
Red potion: 0
Green potion: 0
Blue potion: 0
Pink potion: 0
Cyan potion: 0
Yellow potion: 0
Health: 9.0
Food: 9
Drink: 9
Energy: 9
Mana: 9
XP: 0
Dexterity: 1
Strength: 1
Intelligence: 1
Direction: right
Light: 0.9910
Is Sleeping: False
Is Resting: False
Learned Fireball: False
Learned Iceball: False
Floor: 0
Ladder Open: True
Is Boss Vulnerable: False
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

--- EXAMPLE 3: Seeking water when thirsty ---
Game State:
Map:
-5, -4: sand
-4, -4: sand
-3, -4: grass
-2, -4: grass
-1, -4: grass
0, -4: grass
1, -4: grass
2, -4: grass
3, -4: tree
4, -4: grass
5, -4: grass
-5, -3: sand
-4, -3: sand
-3, -3: tree
-2, -3: grass
-1, -3: grass
0, -3: grass
1, -3: grass
2, -3: grass
3, -3: grass
4, -3: Skeleton on grass
5, -3: grass
-5, -2: water
-4, -2: sand
-3, -2: grass
-2, -2: grass
-1, -2: grass
0, -2: water
1, -2: grass
2, -2: grass
3, -2: grass
4, -2: grass
5, -2: grass
-5, -1: water
-4, -1: sand
-3, -1: grass
-2, -1: grass
-1, -1: grass
0, -1: grass
1, -1: grass
2, -1: grass
3, -1: grass
4, -1: grass
5, -1: grass
-5, 0: water
-4, 0: sand
-3, 0: grass
-2, 0: grass
-1, 0: grass
0, 0: grass
1, 0: grass
2, 0: grass
3, 0: grass
4, 0: grass
5, 0: grass
-5, 1: water
-4, 1: sand
-3, 1: grass
-2, 1: grass
-1, 1: grass
0, 1: grass
1, 1: grass
2, 1: grass
3, 1: grass
4, 1: grass
5, 1: path
-5, 2: sand
-4, 2: sand
-3, 2: grass
-2, 2: grass
-1, 2: grass
0, 2: grass
1, 2: grass
2, 2: tree
3, 2: stone
4, 2: stone
5, 2: path
-5, 3: sand
-4, 3: sand
-3, 3: crafting_table
-2, 3: grass
-1, 3: grass
0, 3: path
1, 3: path
2, 3: path
3, 3: path
4, 3: stone
5, 3: path
-5, 4: sand
-4, 4: grass
-3, 4: grass
-2, 4: grass
-1, 4: path
0, 4: path
1, 4: path
2, 4: path
3, 4: path
4, 4: stone
5, 4: path

Inventory:
Wood: 2
Stone: 10
Coal: 1
Iron: 2
Diamond: 0
Sapphire: 0
Ruby: 0
Sapling: 1
Torch: 0
Arrow: 0
Book: 0
Stone Pickaxe
Red potion: 0
Green potion: 0
Blue potion: 0
Pink potion: 0
Cyan potion: 0
Yellow potion: 0
Health: 4.0
Food: 5
Drink: 4
Energy: 6
Mana: 9
XP: 0
Dexterity: 1
Strength: 1
Intelligence: 1
Direction: down
Light: 0.8817
Is Sleeping: False
Is Resting: False
Learned Fireball: False
Learned Iceball: False
Floor: 0
Ladder Open: True
Is Boss Vulnerable: False
<think>
I am thirsty (Drink: 4). I have low health (4/10). I need to recover health, so I need to eat and drink water. There is water nearby.
1. Scan Map: Found 'water' at (0, -2).
2. Calculate Position:
   - Target (0, -2). Me (0, 0).
   - Delta: (0, -2).
   - Analysis: 2 steps LEFT.
3. Plan: This is a simple situation with a clear single objective: reach the water. No threats nearby. I will move LEFT.
</think>
**Action:** 1 (LEFT)

--- EXAMPLE 4: Complex strategy - Dodging an arrow to reach food ---
Game State:
Map:
-5, -4: water
-4, -4: water
-3, -4: water
-2, -4: sand
-1, -4: grass
0, -4: grass
1, -4: grass
2, -4: grass
3, -4: grass
4, -4: grass
5, -4: grass
-5, -3: water
-4, -3: water
-3, -3: water
-2, -3: sand
-1, -3: grass
0, -3: grass
1, -3: grass
2, -3: grass
3, -3: grass
4, -3: grass
5, -3: grass
-5, -2: water
-4, -2: water
-3, -2: water
-2, -2: sand
-1, -2: sand
0, -2: grass
1, -2: grass
2, -2: grass
3, -2: grass
4, -2: grass
5, -2: grass
-5, -1: water
-4, -1: water
-3, -1: water
-2, -1: sand
-1, -1: sand
0, -1: grass
1, -1: grass
2, -1: grass
3, -1: grass
4, -1: grass
5, -1: grass
-5, 0: water
-4, 0: water
-3, 0: water
-2, 0: sand
-1, 0: sand
0, 0: grass
1, 0: grass
2, 0: grass
3, 0: grass
4, 0: grass
5, 0: grass
-5, 1: water
-4, 1: water
-3, 1: water
-2, 1: sand
-1, 1: sand
0, 1: grass
1, 1: Cow
2, 1: grass
3, 1: grass
4, 1: grass
5, 1: grass
-5, 2: water
-4, 2: water
-3, 2: water
-2, 2: sand
-1, 2: sand
0, 2: Arrow
1, 2: grass
2, 2: grass
3, 2: grass
4, 2: grass
5, 2: grass
-5, 3: water
-4, 3: iron
-3, 3: stone
-2, 3: path
-1, 3: stone
0, 3: Skeleton
1, 3: stone
2, 3: stone
3, 3: stone
4, 3: grass
5, 3: tree
-5, 4: water
-4, 4: stone
-3, 4: stone
-2, 4: path
-1, 4: stone
0, 4: stone
1, 4: stone
2, 4: stone
3, 4: stone
4, 4: stone
5, 4: grass

Inventory:
Wood: 6
Stone: 14
Coal: 1
Iron: 2
Diamond: 0
Sapphire: 0
Ruby: 0
Sapling: 1
Torch: 0
Arrow: 0
Book: 0
Stone Pickaxe
Red potion: 0
Green potion: 0
Blue potion: 0
Pink potion: 0
Cyan potion: 0
Yellow potion: 0
Health: 7.0
Food: 2
Drink: 6
Energy: 4
Mana: 9
XP: 0
Dexterity: 1
Strength: 1
Intelligence: 1
Direction: down
Light: 0.1145
Is Sleeping: False
Is Resting: False
Learned Fireball: False
Learned Iceball: False
Floor: 0
Ladder Open: True
Is Boss Vulnerable: False
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
</think>
**Action:** 4 (DOWN)

--- EXAMPLE 5: Strategic decision to fight ---
Game State:
Map:
-5, -4: Darkness
-4, -4: Darkness
-3, -4: Darkness
-2, -4: wall
-1, -4: path
0, -4: wall
1, -4: darkness
2, -4: darkness
3, -4: darkness
4, -4: darkness
5, -4: darkness
-5, -3: Darkness
-4, -3: Darkness
-3, -3: Darkness
-2, -3: wall
-1, -3: path
0, -3: wall
1, -3: darkness
2, -3: darkness
3, -3: darkness
4, -3: darkness
5, -3: darkness
-5, -2: Darkness
-4, -2: Darkness
-3, -2: Darkness
-2, -2: wall
-1, -2: path
0, -2: wall
1, -2: darkness
2, -2: darkness
3, -2: darkness
4, -2: darkness
5, -2: darkness
-5, -1: Darkness
-4, -1: Darkness
-3, -1: Darkness
-2, -1: wall
-1, -1: path
0, -1: wall_moss
1, -1: wall
2, -1: wall
3, -1: wall
4, -1: wall_moss
5, -1: wall
-5, 0: Darkness
-4, 0: Darkness
-3, 0: Darkness
-2, 0: wall
-1, 0: path
0, 0: path
1, 0: Orc Soldier on torch on path
2, 0: path
3, 0: path
4, 0: Snail on path
5, 0: path
-5, 1: Darkness
-4, 1: Darkness
-3, 1: Darkness
-2, 1: wall
-1, 1: path
0, 1: wall
1, 1: path
2, 1: path
3, 1: path
4, 1: path
5, 1: path
-5, 2: Darkness
-4, 2: Darkness
-3, 2: Darkness
-2, 2: wall
-1, 2: path
0, 2: wall
1, 2: path
2, 2: path
3, 2: path
4, 2: path
5, 2: path
-5, 3: Darkness
-4, 3: Darkness
-3, 3: Darkness
-2, 3: wall
-1, 3: path
0, 3: wall
1, 3: path
2, 3: path
3, 3: path
4, 3: path
5, 3: chest
-5, 4: Darkness
-4, 4: Darkness
-3, 4: Darkness
-2, 4: wall
-1, 4: path
0, 4: wall
1, 4: path
2, 4: path
3, 4: path
4, 4: fountain
5, 4: path

Inventory:
Wood: 17
Stone: 61
Coal: 0
Iron: 4
Diamond: 1
Sapphire: 0
Ruby: 0
Sapling: 2
Torch: 0
Arrow: 0
Book: 0
Iron Pickaxe
Iron Sword with No enchantment
Red potion: 0
Green potion: 0
Blue potion: 0
Pink potion: 0
Cyan potion: 0
Yellow potion: 0
Iron Helmet with No enchantment
Health: 6.3
Food: 3
Drink: 8
Energy: 8
Mana: 9
XP: 1
Dexterity: 1
Strength: 1
Intelligence: 1
Direction: down
Light: 0.4222
Is Sleeping: False
Is Resting: False
Learned Fireball: False
Learned Iceball: False
Floor: 1
Ladder Open: False
Is Boss Vulnerable: False
<think>
An Orc Soldier is blocking my path. I have decent health (7.0) and Iron gear (Sword, Armor). I am confident I can win.
1. Scan Map: 'Orc Soldier' at (1, 0).
2. Calculate Position:
   - Target (1, 0) - Me (0, 0) = (+1 Row, 0 Col).
   - Analysis: The enemy is 1 step DOWN.
3. Orientation Check: My Direction is 'down'.
   - Result: I am facing the enemy.
4. Strategy: Fight. I have armor and a sword. I will attack to clear the path and potentially get loot.
Action: DO (Attack).
</think>
**Action:** 5 (DO)

--- END OF EXAMPLES ---
==================================================
>>> LIVE ENVIRONMENT STREAM STARTS HERE <<<
>>> IGNORE ALL COORDINATES FROM EXAMPLES ABOVE <<<
==================================================
"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Below are examples of good gameplay decisions. These are EXAMPLES ONLY, not your actual game history:\n{few_shot_examples}\nYou are at (0,0). Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>).\n\nYOUR CURRENT GAME STATE (use ONLY this map for coordinates):\n{text_obs}"},
    ]
    
    # Captures the EXACT string passed to the model (System + User)
    full_prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(full_prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # 2. First Generation Pass (Thinking)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)
    
    output_ids = outputs[0].tolist()
    patched = False

    # 3. decode full response to check for action
    full_response_text = processor.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
    
    # 4. Try to find action in the initial response
    action = 0
    # Allow optional asterisks to be robust to minor formatting errors
    action_pattern = r'(?:\*\*|)?Action:(?:\*\*|)?\s*(\d+)'
    
    # Find ALL matches and take the LAST one to allow for chain-of-thought self-corrections
    matches = list(re.finditer(action_pattern, full_response_text, re.IGNORECASE))
    
    found_valid_action = False
    if matches:
        last_match = matches[-1]
        try:
            val = int(last_match.group(1))
            if 0 <= val <= 42:
                action = val
                found_valid_action = True
        except ValueError:
            pass

    # 5. Intervention: If no valid action found, FORCE it.
    patched = False
    if not found_valid_action:
        patched = True
        # print("--- Action missing/invalid. Patching... ---")
        
        # We append the trigger string to FORCE the model to output an ID
        # Check if </think> was already generated
        current_context_text = processor.decode(output_ids, skip_special_tokens=True)
        if "</think>" not in current_context_text:
             forced_extension = "\n</think>\nOk I need to output an action now. \n**Action:** "
        else:
             forced_extension = "\nOk I need to output an action now. \n**Action:** "
        
        # Re-tokenize context + extension
        current_context_text = processor.decode(output_ids, skip_special_tokens=True)
        new_full_text = current_context_text + forced_extension
        new_inputs = processor(new_full_text, return_tensors="pt").to(model.device)
        
        # Generate JUST the action ID (short max_new_tokens)
        with torch.no_grad():
            final_outputs = model.generate(
                **new_inputs, 
                max_new_tokens=5, 
                do_sample=True, 
                temperature=0.7
            )
        output_ids = final_outputs[0].tolist()
        
        # Re-decode everything with the extension
        full_response_text = processor.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
        
        # Attempt extraction again on the FORCED text
        matches = list(re.finditer(action_pattern, full_response_text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            try:
                val = int(last_match.group(1))
                if 0 <= val <= 42:
                    action = val
            except ValueError:
                pass

    # 6. Parse Content for Logging
    try:
        rev_index = output_ids[::-1].index(SPLIT_TOKEN_ID)
        split_index = len(output_ids) - rev_index
        thinking_content = processor.decode(output_ids[prompt_len:split_index], skip_special_tokens=True).strip()
        content = processor.decode(output_ids[split_index:], skip_special_tokens=True).strip()
    except ValueError:
        thinking_content = full_response_text
        content = ""

    log_data = {
        "action": action,
        "is_patched": patched,
        "full_prompt": full_prompt_text,
        "full_response": full_response_text,
        "thinking": thinking_content,
        "content": content
    }
    
    return action, log_data

def render_frame_for_video(state, step_num, action_name, reward, total_reward):
    """Render a frame with game pixels and overlay info."""
    pixels = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_AGENT, do_night_noise=False)
    frame = np.array(pixels, dtype=np.uint8)

    # Resize for better visibility
    scale = 4
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Add info overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    overlay_height = 60
    frame_with_overlay = np.zeros((frame.shape[0] + overlay_height, frame.shape[1], 3), dtype=np.uint8)
    frame_with_overlay[overlay_height:, :] = frame

    cv2.putText(frame_with_overlay, f"Step: {step_num}", (10, 15), font, font_scale, color, thickness)
    cv2.putText(frame_with_overlay, f"Action: {action_name}", (10, 35), font, font_scale, color, thickness)
    cv2.putText(frame_with_overlay, f"Reward: {reward:.2f} | Total: {total_reward:.2f}", (10, 55), font, font_scale, color, thickness)

    return frame_with_overlay

def get_achieved_achievements(state) -> list[str]:
    achievements_array = np.array(state.achievements)
    achieved = []
    for achievement in Achievement:
        if achievements_array[achievement.value] > 0:
            achieved.append(achievement.name)
    return achieved

def create_action_distribution_chart(action_counts: Counter) -> np.ndarray:
    action_names = [Action(i).name for i in range(43)]
    counts = [action_counts.get(i, 0) for i in range(43)]
    used_actions = [(name, count) for name, count in zip(action_names, counts) if count > 0]
    if not used_actions:
        used_actions = [("NOOP", 0)]
    names, values = zip(*used_actions)

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title('Action Distribution')
    fig.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img

def render_craftax_text_swapped(state):
    # Standard renderer outputs "Col, Row" (x, y)
    # We want "Row, Col" (y, x) to match the prompt and LLM reasoning.
    st = render_craftax_text(state)
    lines = st.split('\n')
    new_lines = []
    
    # Regex to match the coordinate line: "Col, Row: Object"
    # We want "Row, Col: Object"
    # Matches start of line like "-1, 0: " or "0, 0: "
    coord_pattern = re.compile(r"^(-?\d+),\s*(-?\d+):")
    
    for line in lines:
        match = coord_pattern.match(line)
        if match:
            col, row = match.groups()
            # Swap them
            new_line = line.replace(f"{col}, {row}:", f"{row}, {col}:", 1)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
            
    return '\n'.join(new_lines)


def main():
    parser = argparse.ArgumentParser(description="LLM agent playing Craftax")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507", help="Model to use")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument("--wandb_project", type=str, default="craftax-llm-play", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--save_video", action="store_true", help="Save video locally")
    parser.add_argument("--video_dir", type=str, default="./llm_play_videos", help="Directory to save videos")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    args = parser.parse_args()

    # Initialize WandB
    run_name = args.run_name or f"llm-play-seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_id": args.model,
            "seed": args.seed,
            "max_steps": args.max_steps,
        },
    )
    print(f"[WandB] Initialized run: {run_name}")

    model, processor = load_model(args.model)
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)

    total_reward = 0.0
    step = 0
    action_counts = Counter()
    llm_responses = [] # Store dicts for the table
    frames = []

    print("\n" + "="*60)
    print("Starting Craftax with llm agent")
    print("="*60)

    # Define columns for WandB table
    columns = [
        "Step", 
        "Full Prompt",
        "Action ID", 
        "Action Name", 
        "Patched", 
        "Thinking", 
        "Final Answer", 
        "Full Response"
    ]

    # Initialize Table with log_mode="MUTABLE"
    # columns defined above
    llm_table = wandb.Table(columns=columns, log_mode="MUTABLE")  
    while step < args.max_steps:
        # Get text obs with SWAPPED coordinates (Row, Col)
        text_obs = render_craftax_text_swapped(state)
        
        action, log_data = get_action(model, processor, text_obs)
        action_name = Action(action).name

        action_counts[action] += 1

        llm_table.add_data(
            step, 
            log_data["full_prompt"],
            action, 
            action_name, 
            log_data["is_patched"], 
            log_data["thinking"],
            log_data["content"],
            log_data["full_response"]
        )
        
        # Append to llm_responses for final table
        llm_responses.append({
            "step": step,
            "action": action,
            "action_name": action_name,
            "is_patched": log_data["is_patched"],
            "thinking": log_data["thinking"],
            "content": log_data["content"],
            "full_response": log_data["full_response"],
            "full_prompt": log_data["full_prompt"],
        })

        rng, step_rng = jax.random.split(rng)
        old_achievements = np.array(state.achievements).copy()
        obs, state, reward, done, info = env.step(step_rng, state, action, env_params)
        new_achievements = np.array(state.achievements)

        total_reward += float(reward)
        step += 1

        frame = render_frame_for_video(state, step, action_name, float(reward), total_reward)
        frames.append(frame)

        step_log = {
            "step": step,
            "action": action,
            "reward": float(reward),
            "total_reward": total_reward,
            "player_health": float(state.player_health),
            "player_food": float(state.player_food),
            "player_drink": float(state.player_drink),
            "player_energy": float(state.player_energy),
            "patched_action": int(log_data["is_patched"]), # Scalar metric for quick graphing
        }

        for achievement in Achievement:
            if new_achievements[achievement.value] > old_achievements[achievement.value]:
                step_log[f"achievement/{achievement.name}"] = 1
                print(f"  [Achievement] {achievement.name} earned!")

        if step % 10 == 0:
            step_log["detailed_logs"] = llm_table
            
        if step == 10:
            # Log intermediate table at step 10
            temp_table = wandb.Table(columns=columns)
            for resp in llm_responses:
                temp_table.add_data(
                    resp["step"], 
                    resp["full_prompt"],
                    resp["action"], 
                    resp["action_name"], 
                    resp["is_patched"], 
                    resp["thinking"],
                    resp["content"],
                    resp["full_response"]
                )
            wandb.log({"detailed_logs_step_10": temp_table})
            print("Logged intermediate table at step 10")
            
        wandb.log(step_log)

        if step == 0:
            print(f"Step {step}: {action_name} (action={action}), Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"  Thinking: {log_data['thinking']}...\n\n") 
            print(f"  Full Response: {log_data['full_response']}\n\n")

        if done:
            print(f"\n{'='*60}")
            print(f"Episode ended after {step} steps. Total reward: {total_reward:.2f}")
            print(f"{'='*60}")
            break

    final_achievements = get_achieved_achievements(state)
    print(f"\nAchievements earned ({len(final_achievements)}):")
    for ach in final_achievements:
        print(f"  - {ach}")

    summary_log = {
        "episode/total_reward": total_reward,
        "episode/length": step,
        "episode/num_achievements": len(final_achievements),
        "episode/achievements": final_achievements,
    }

    try:
        action_chart = create_action_distribution_chart(action_counts)
        summary_log["episode/action_distribution_chart"] = wandb.Image(action_chart, caption="Action Distribution")
    except Exception as e:
        print(f"Warning: Failed to create action chart: {e}")

    for action_id, count in action_counts.items():
        summary_log[f"action_counts/{Action(action_id).name}"] = count

    for achievement in Achievement:
        achieved = 1 if achievement.name in final_achievements else 0
        summary_log[f"final_achievements/{achievement.name}"] = achieved

    wandb.log(summary_log)

    # Define columns to include all the new data
    # Define columns to include all the new data
    # (columns defined at start of main)
    llm_table = wandb.Table(columns=columns)
    
    for resp in llm_responses:
        llm_table.add_data(
            resp["step"], 
            resp["full_prompt"],
            resp["action"], 
            resp["action_name"], 
            resp["is_patched"], 
            resp["thinking"],
            resp["content"],
            resp["full_response"]
        )
        
    wandb.log({"detailed_logs": llm_table})
    print("Detailed logs table uploaded to WandB.")

    if len(frames) > 0:
        if args.save_video:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"llm_play_seed{args.seed}.mp4"

            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"Video saved locally: {video_path}")

        video_array = np.array(frames) 
        video_array = np.transpose(video_array, (0, 3, 1, 2)) 
        wandb.log({
            "episode/video": wandb.Video(video_array, fps=15, format="mp4")
        })
        print("Video logged to WandB")

    print("\nAction Distribution:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
    for action_id, count in sorted_actions[:10]: 
        pct = count / step * 100
        print(f"  {Action(action_id).name}: {count} ({pct:.1f}%)")

    wandb.finish()
    print("\nWandB run finished.")

if __name__ == "__main__":
    main()
