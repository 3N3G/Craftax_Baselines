# Craftax LLM Player Prompt Documentation

This document contains the system prompt and guidance for LLM agents playing Craftax. This is used by `vlm_play.py` to configure the Qwen3-4B-Thinking-2507 model.

## Game Overview

Craftax is a procedurally-generated Minecraft-inspired gridworld game about exploring dungeons, mining resources, crafting tools, and fighting enemies. The player navigates a 2D grid world with the goal of descending through 9 increasingly difficult dungeon levels to reach and defeat the final boss.

## Core Game Mechanics

### Coordinate System

**IMPORTANT**: Craftax uses an unusual coordinate system:
- Position is stored as `[x, y]` where:
  - `x` = row index (vertical axis)
  - `y` = column index (horizontal axis)

**Direction mappings:**
- **UP**: `[-1, 0]` (x decreases)
- **DOWN**: `[1, 0]` (x increases)  
- **LEFT**: `[0, -1]` (y decreases)
- **RIGHT**: `[0, 1]` (y increases)

**Example**: Position `(2, -2)` from origin is:
- x = 2 → DOWN 2 steps
- y = -2 → LEFT 2 steps

### Player Movement and Interaction

The player can:
- **Move** in four cardinal directions (UP, DOWN, LEFT, RIGHT)
- **Interact (DO action)** with the environment, which contextually performs different actions:
  - **Mine** blocks (trees, stone, ore deposits)
  - **Attack** creatures
  - **Drink** from water tiles or fountains
  - **Eat** fruit from plants
  - **Open** treasure chests
  - **IMPORTANT**: Interaction only works if the block/creature/chest is directly in front of the player (one step in the direction they're facing)

### Player Intrinsics (Survival Stats)

The player has 5 vital statistics that must be managed:

1. **Health**: Player dies if this reaches 0
   - Recovers when hunger, thirst, and energy are all above 0
   - Decreases when any of hunger/thirst/energy is at 0
   - Lost when taking damage from creatures

2. **Hunger**: Depletes naturally over time
   - Replenished by eating (meat from animals, fruit from plants)
   - If at 0, health will start decreasing

3. **Thirst**: Depletes naturally over time
   - Replenished by drinking water or from fountains
   - If at 0, health will start decreasing

4. **Energy**: Depletes naturally over time
   - Replenished by sleeping or resting
   - If at 0, health will start decreasing

5. **Mana**: Used for casting spells and enchanting items
   - Recovers naturally over time
   - Required for fireball, iceball spells and enchantments

### Progression System

**CRITICAL**: To progress through the game, the player must:

1. **Find the ladder** on each floor
2. **Open the ladder** by killing 8 creatures on that level (except overworld where ladder starts open)
3. **Descend** to the next level
4. Repeat for all 9 levels until reaching the final boss

Each floor has unique challenges and increasingly difficult creatures. The game becomes progressively harder as you descend.

### Resource Gathering and Crafting

The core gameplay loop involves gathering resources and crafting better equipment:

#### Resources
- **Wood**: From trees (use DO action on trees)
- **Stone**: From stone blocks
- **Coal**: Mining coal ore deposits
- **Iron**: Mining iron ore deposits  
- **Diamond**: Mining diamond ore (deeper levels)
- **Sapphire, Ruby**: Rare gems from mining

#### Tool/Weapon Progression
1. **Wood tools** (first tier) - requires wood
   - Wood pickaxe (for mining stone)
   - Wood sword (for fighting weak creatures)
   
2. **Stone tools** (second tier) - requires stone
   - Stone pickaxe (for mining iron ore)
   - Stone sword (better combat)
   
3. **Iron tools** (third tier) - requires iron ore + furnace
   - Iron pickaxe (for mining diamond ore)
   - Iron sword (strong combat)
   - Iron armor (damage reduction)
   
4. **Diamond tools** (top tier) - requires diamond
   - Diamond pickaxe (mines anything)
   - Diamond sword (maximum damage)
   - Diamond armor (best protection)

#### Crafting Dependencies
- **Crafting Table**: Requires 2 wood to place, essential for making all tools
- **Furnace**: Requires placing stone blocks, needed for smelting iron/diamond
- **Torches**: Craft from wood + coal, essential for dungeon exploration (provides light)

### Combat System

- Use the **DO action** when facing a creature to attack
- Consider your equipment:
  - Better swords deal more damage
  - Armor reduces damage taken
- Can also use:
  - **Arrows** (SHOOT_ARROW): Ranged attacks
  - **Spells** (CAST_FIREBALL, CAST_ICEBALL): Magical attacks requiring mana

### Exploration and Vision

- **Light level** affects visibility
- **Torches** are critical for dungeon levels (dark without them)
- Use PLACE_TORCH to illuminate areas
- Map shows surroundings in an 11x11 grid centered on player

## Recommended Progression Strategy

Here is a robust strategy for progressing through Craftax:

### Phase 1: Early Game (Overworld - Floor 0)

1. **Gather wood** from nearby trees (DO action on trees)
2. **Place a crafting table** (PLACE_TABLE action requires 2 wood)
3. **Make wood tools**:
   - Wood pickaxe (MAKE_WOOD_PICKAXE)
   - Wood sword (MAKE_WOOD_SWORD)
4. **Mine stone** using wood pickaxe (DO action on stone blocks)
5. **Place furnace** from stone (PLACE_FURNACE)
6. **Make stone tools**:
   - Stone pickaxe (MAKE_STONE_PICKAXE)
   - Stone sword (MAKE_STONE_SWORD)

### Phase 2: Iron Age (Overworld - Floor 0)

1. **Mine iron ore** using stone pickaxe
2. **Gather coal** from ore deposits (for smelting and torches)
3. **Smelt iron** at furnace
4. **Make iron equipment**:
   - Iron pickaxe (MAKE_IRON_PICKAXE)
   - Iron sword (MAKE_IRON_SWORD)  
   - Iron armor (MAKE_IRON_ARMOUR)
5. **Craft torches** (MAKE_TORCH) - need wood + coal
6. **Stock up on supplies**:
   - Extra coal and wood for more torches
   - Food (kill cows with DO action, gather meat)
   - Water nearby for drinking

### Phase 3: Dungeon Exploration (Floors 1-8)

1. **Locate the ladder** on each floor
2. **Kill 8 creatures** to open the ladder
   - Maintain health by:
     - Killing peaceful animals for food
     - Drinking water when available
     - Resting (REST action) to restore energy
3. **Descend** when ladder opens (DESCEND action when standing on ladder)
4. **Place torches** regularly to maintain visibility
5. **Upgrade to diamond equipment** when possible:
   - Mine diamond ore on deeper floors
   - Diamond pickaxe (MAKE_DIAMOND_PICKAXE)
   - Diamond sword (MAKE_DIAMOND_SWORD)
   - Diamond armor (MAKE_DIAMOND_ARMOUR)

### Phase 4: End Game (Floor 9 - Boss Level)

1. Ensure you have:
   - Best equipment (diamond tier)
   - Full health and resources
   - Sufficient torches
2. Defeat the final boss

## Survival Tips

### Managing Intrinsics

- **Monitor all 5 stats constantly** (health, hunger, thirst, energy, mana)
- **Keep intrinsics above 0** - if hunger/thirst/energy hit 0, health decreases
- **Prioritize drinking and eating** - easier than recovering from low health
- **Rest when energy is low** (REST action is faster than SLEEP)
- **Kill peaceful creatures** (cows, etc.) for reliable food sources

### Resource Management

- **Stockpile coal and wood** for torches before descending
- **Carry extra resources** for emergency crafting
- **Don't waste iron/diamond** - only craft when needed

### Combat Strategy

- **Upgrade weapons before descending** to new floors
- **Avoid unnecessary fights** if low on health
- **Use armor** - craft it before entering dangerous areas
- **Kite enemies** - attack and retreat to minimize damage

### Efficient Exploration

- **Systematic search patterns** to find ladders quickly
- **Mark explored areas mentally** to avoid wasting time
- **Use torches liberally** in dungeons for better visibility

## Available Actions

The model must output a number from 0-42 corresponding to these actions:

```
0:  NOOP            - Do nothing
1:  LEFT            - Move left
2:  RIGHT           - Move right  
3:  UP              - Move up
4:  DOWN            - Move down
5:  DO              - Interact (mine/attack/drink/eat/open)
6:  SLEEP           - Sleep (slowly restores energy)
7:  PLACE_STONE     - Place a stone block
8:  PLACE_TABLE     - Place crafting table
9:  PLACE_FURNACE   - Place furnace
10: PLACE_PLANT     - Place a plant (from sapling)
11: MAKE_WOOD_PICKAXE   - Craft wood pickaxe
12: MAKE_STONE_PICKAXE  - Craft stone pickaxe
13: MAKE_IRON_PICKAXE   - Craft iron pickaxe
14: MAKE_WOOD_SWORD     - Craft wood sword
15: MAKE_STONE_SWORD    - Craft stone sword
16: MAKE_IRON_SWORD     - Craft iron sword
17: REST                - Rest (quickly restores energy)
18: DESCEND             - Go down ladder
19: ASCEND              - Go up ladder
20: MAKE_DIAMOND_PICKAXE - Craft diamond pickaxe
21: MAKE_DIAMOND_SWORD   - Craft diamond sword
22: MAKE_IRON_ARMOUR     - Craft iron armor
23: MAKE_DIAMOND_ARMOUR  - Craft diamond armor
24: SHOOT_ARROW          - Fire an arrow
25: MAKE_ARROW           - Craft arrows
26: CAST_FIREBALL        - Cast fireball spell
27: CAST_ICEBALL         - Cast iceball spell
28: PLACE_TORCH          - Place a torch
29: DRINK_POTION_RED     - Drink red potion
30: DRINK_POTION_GREEN   - Drink green potion
31: DRINK_POTION_BLUE    - Drink blue potion
32: DRINK_POTION_PINK    - Drink pink potion
33: DRINK_POTION_CYAN    - Drink cyan potion
34: DRINK_POTION_YELLOW  - Drink yellow potion
35: READ_BOOK            - Read a book
36: ENCHANT_SWORD        - Enchant sword
37: ENCHANT_ARMOUR       - Enchant armor
38: MAKE_TORCH           - Craft torches (requires wood + coal)
39: LEVEL_UP_DEX         - Level up dexterity
40: LEVEL_UP_STR         - Level up strength
41: LEVEL_UP_INT         - Level up intelligence
42: ENCHANT_BOW          - Enchant bow
```

## System Prompt Template

The following is the complete system prompt used for the Qwen model:

```
You are playing Craftax.

Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions and can interact. Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest.

The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. Mana is used for casting spells or enchanting items and will naturally recover. Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. If the players health falls beneath 0 they will die and the game will restart.

IMPORTANT: To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). There are 9 levels in total.

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
```

## User Message Template

Format for each game state observation:

```
Current game state:
Map:
[coordinate pairs with terrain/entity information]

Inventory:
[list of items and quantities]
[intrinsics: Health, Food, Drink, Energy, Mana]
[player stats: XP, Dexterity, Strength, Intelligence]
[state info: Direction, Light level, Is Sleeping, Is Resting, Floor number, etc.]

You are at (0,0); pick an action. Think about the scene briefly, enough to pick a move and then say the number. Here are some examples of good gameplay:

[FEW-SHOT EXAMPLES GO HERE]
```

## Few-Shot Examples

TODO: Add 3-5 high-quality example trajectories showing:
- Early game wood gathering
- Mining and crafting progression
- Survival intrinsic management
- Combat with creatures
- Ladder location and descending

Each example should show:
1. Game state observation
2. Agent reasoning in `<think>` tags
3. Action selection with number

Example format:
```
<|im_start|>user
Current game state:
[map and inventory]
You are at (0,0); pick an action.
<|im_end|>
<|im_start|>assistant
<think>
I see trees nearby at (-5,-3) and (-3,2). I need wood to start crafting. I should move toward the nearest tree and use the DO action to gather wood.
</think>
Action: 1 (LEFT to move toward tree)
<|im_end|>
```

## Model Output Format

The Qwen model is expected to output in this format:

```
<think>
[Internal reasoning about the current game state, strategy, and action selection]
</think>

Action: [number from 0-42]
```

The `vlm_play.py` script extracts the number from the response using regex.

## Implementation Notes for vlm_play.py

- Model: Qwen/Qwen3-4B-Thinking-2507
- Configured with `<think>` tags for chain-of-thought reasoning
- Timeout handling: If model exceeds token limit during thinking, a patching mechanism forces completion
- Action extraction: Regex search for valid action numbers (0-42) in model output
- Temperature: 0.7 (allows some exploration while maintaining coherent reasoning)
- Max tokens for thinking: 2048
- Max tokens for final answer: 128
