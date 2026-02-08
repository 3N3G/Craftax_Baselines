# Rendering Symbolic Observations to Pixels

For policy evaluation with visualizations, symbolic observations can be rendered to pixels.

## Key Point

Symbolic observations are **flattened vectors** (shape ~8268). To render pixels, you need the **game state object** (`EnvState`), not the observation.

## How to Render

```python
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name

# Create symbolic env
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

# Step through env - keep the state
obs, state = env.reset(rng, env_params)
obs, state, reward, done, info = env.step(rng, state, action, env_params)

# Render state to pixels
pixels = render_craftax_pixels(state)  # Returns image array
```

## For Evaluation Scripts

When evaluating a trained policy:

1. Load the policy (trained on symbolic obs)
2. Run the policy: `action = policy(symbolic_obs)`
3. Step the env: `obs, state, reward, done, info = env.step(...)`
4. Render for visualization: `pixels = render_craftax_pixels(state)`
5. Save pixels to video/images

## Example Usage (eval loop)

```python
frames = []
obs, state = env.reset(rng, env_params)

for step in range(max_steps):
    action = policy(obs)  # Policy uses symbolic obs
    obs, state, reward, done, info = env.step(rng, state, action, env_params)
    
    # Render for video
    frame = render_craftax_pixels(state)
    frames.append(frame)
    
    if done:
        break

# Save frames to video
imageio.mimsave("eval_rollout.mp4", frames, fps=10)
```

## See Also

- `vlm_play.py` - Uses `render_craftax_pixels` for video generation
- `render_craftax_text` - Converts state to text description
