#!/usr/bin/env python3
"""Simple script to query Qwen3-4B with a Craftax observation.

Usage:
    python query_qwen.py "Map (interesting tiles only): -1, 0:tree, 0, 1:stone"
    python query_qwen.py --file observation.txt
    python query_qwen.py  # Interactive mode
"""

import argparse
import sys
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen3-4B"

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


def load_model(model_id: str = None):
    """Load a Qwen model."""
    model_id = model_id or DEFAULT_MODEL
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"Model loaded! Hidden size: {model.config.hidden_size}")
    return model, processor


def query_model(model, processor, observation: str, temperature: float = 0.7, max_tokens: int = 2048) -> dict:
    """
    Query the model with a Craftax observation.
    
    Args:
        model: The loaded model
        processor: The tokenizer/processor  
        observation: The text observation (filtered or raw)
        temperature: Sampling temperature
        max_tokens: Maximum new tokens to generate
    
    Returns:
        dict with 'full_prompt', 'full_response', 'thinking', 'content'
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Below are examples of good gameplay decisions. These are EXAMPLES ONLY, not your actual game history:\n{FEW_SHOT_EXAMPLES}\nYOUR CURRENT GAME STATE (use ONLY this map for coordinates):\n{observation}\n\nYou are at (0,0). Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>)."},
    ]
    
    full_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(full_prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=True, 
            temperature=temperature
        )
    
    output_ids = outputs[0].tolist()
    full_response = processor.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
    
    # Try to split thinking from content
    SPLIT_TOKEN_ID = 151668  # </think>
    try:
        rev_index = output_ids[::-1].index(SPLIT_TOKEN_ID)
        split_index = len(output_ids) - rev_index
        thinking = processor.decode(output_ids[prompt_len:split_index], skip_special_tokens=True).strip()
        content = processor.decode(output_ids[split_index:], skip_special_tokens=True).strip()
    except ValueError:
        thinking = full_response
        content = ""
    
    return {
        "full_prompt": full_prompt,
        "full_response": full_response,
        "thinking": thinking,
        "content": content,
    }


def interactive_loop(model, processor, temperature: float, max_tokens: int, show_prompt: bool):
    """Interactive REPL mode - model stays loaded between queries."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE - Model loaded, ready for queries")
    print("="*60)
    print("Commands:")
    print("  :q or :quit  - Exit")
    print("  :f <file>    - Load observation from file")
    print("  :t <temp>    - Set temperature (current: {})".format(temperature))
    print("  Otherwise, paste observation and press Enter twice (blank line) to submit")
    print("="*60 + "\n")
    
    while True:
        print("\n>>> Enter observation (blank line to submit, :q to quit):")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            
            # Check for commands
            if not lines and line.strip().startswith(':'):
                cmd = line.strip()
                if cmd in (':q', ':quit'):
                    print("Exiting...")
                    return
                elif cmd.startswith(':f '):
                    filepath = cmd[3:].strip()
                    try:
                        with open(filepath, 'r') as f:
                            observation = f.read().strip()
                        lines = [observation]
                        break
                    except Exception as e:
                        print(f"Error reading file: {e}")
                        continue
                elif cmd.startswith(':t '):
                    try:
                        temperature = float(cmd[3:].strip())
                        print(f"Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid temperature value")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Blank line = submit
            if line == "" and lines:
                break
            lines.append(line)
        
        if not lines:
            continue
            
        observation = "\n".join(lines)
        
        print(f"\n{'='*60}")
        print("OBSERVATION:")
        print('='*60)
        print(observation[:500] + "..." if len(observation) > 500 else observation)
        print('='*60)
        print("\nQuerying model...")
        
        result = query_model(model, processor, observation, temperature, max_tokens)
        
        if show_prompt:
            print("\n" + "="*60)
            print("FULL PROMPT:")
            print("="*60)
            print(result["full_prompt"])
        
        print("\n" + "="*60)
        print("FULL RESPONSE:")
        print("="*60)
        print(result["full_response"])


def main():
    parser = argparse.ArgumentParser(description="Query Qwen with a Craftax observation")
    parser.add_argument("observation", nargs="?", default=None, help="The observation text (or use --file)")
    parser.add_argument("--file", "-f", type=str, help="Read observation from file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode - keep model loaded for multiple queries")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens")
    parser.add_argument("--show_prompt", action="store_true", help="Print the full prompt")
    args = parser.parse_args()
    
    # Load model first for interactive mode or single query
    model, processor = load_model(args.model)
    
    # Interactive mode
    if args.interactive:
        interactive_loop(model, processor, args.temperature, args.max_tokens, args.show_prompt)
        return
    
    # Get observation for single query
    if args.file:
        with open(args.file, 'r') as f:
            observation = f.read().strip()
    elif args.observation:
        observation = args.observation
    else:
        print("Enter observation (Ctrl+D or Ctrl+Z to end):")
        observation = sys.stdin.read().strip()
    
    if not observation:
        print("Error: No observation provided")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("OBSERVATION:")
    print('='*60)
    print(observation)
    print('='*60 + "\n")
    
    result = query_model(model, processor, observation, args.temperature, args.max_tokens)
    
    if args.show_prompt:
        print("\n" + "="*60)
        print("FULL PROMPT:")
        print("="*60)
        print(result["full_prompt"])
    
    print("\n" + "="*60)
    print("FULL RESPONSE:")
    print("="*60)
    print(result["full_response"])
    
    if result["content"]:
        print("\n" + "-"*40)
        print("THINKING:")
        print("-"*40)
        print(result["thinking"])
        print("\n" + "-"*40)
        print("CONTENT:")
        print("-"*40)
        print(result["content"])


if __name__ == "__main__":
    main()
