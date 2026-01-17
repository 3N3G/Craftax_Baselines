import os
import argparse
import torch
import numpy as np
import jax
import wandb
import cv2  # Required for drawing video overlay
import matplotlib.pyplot as plt
from pt_awr import Config, ActorCriticConv
from image_utils import obs_to_01_range, obs_to_255_range

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_platform_name", "cpu")


def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    model = ActorCriticConv(
        action_dim=Config.ACTION_DIM, layer_width=Config.LAYER_WIDTH
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def draw_dual_value_bar(frame, pred_val, true_val, v_min=-1.0, v_max=10.0):
    """
    Draws TWO bars in the footer:
    1. Top: Predicted Value (V)
    2. Bottom: Empirical Return (G)
    """
    # Ensure frame is contiguous numpy array for cv2
    frame = np.ascontiguousarray(frame)
    h, w, c = frame.shape

    # Define footer height (increased to fit two bars)
    footer_h = 50
    new_h = h + footer_h

    # Create new frame with extra height (initialized to black/dark background)
    new_frame = np.zeros((new_h, w, c), dtype=frame.dtype)
    
    # Copy the original scene onto the top of the new frame
    new_frame[0:h, 0:w] = frame

    # --- Shared Config ---
    bar_w = w - 60  # Width of the bar area
    x_start = 55    # Indent to make room for labels
    bg_color = (50, 50, 50)
    text_color = (255, 255, 255)
    
    # Helper to draw a single bar
    def draw_single_bar(y_pos, val, label, color_pos, color_neg):
        # Draw Label
        cv2.putText(
            new_frame, label, (5, y_pos + 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1
        )
        
        # Draw Bar Background
        cv2.rectangle(
            new_frame, (x_start, y_pos), (x_start + bar_w, y_pos + 10), bg_color, -1
        )
        
        # Calculate Fill
        clamped_val = max(v_min, min(v_max, val))
        ratio = (clamped_val - v_min) / (v_max - v_min)
        fill_w = int(ratio * bar_w)
        
        # Determine Color
        c = color_pos if val > 0 else color_neg
        
        # Draw Fill
        if fill_w > 0:
            cv2.rectangle(
                new_frame,
                (x_start, y_pos),
                (x_start + fill_w, y_pos + 10),
                c,
                -1,
            )
        
        # Draw Numeric Text
        cv2.putText(
            new_frame, f"{val:.2f}", (x_start + bar_w + 5, y_pos + 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1
        )

    # --- Draw Predicted Value (Top Bar) ---
    # Green for positive, Red for negative
    draw_single_bar(
        y_pos=h + 10, 
        val=pred_val, 
        label="Pred V", 
        color_pos=(0, 255, 0), 
        color_neg=(0, 0, 255)
    )

    # --- Draw Empirical Return (Bottom Bar) ---
    # Cyan for positive, Magenta for negative (to distinguish from V)
    draw_single_bar(
        y_pos=h + 30, 
        val=true_val, 
        label="True G", 
        color_pos=(255, 255, 0), # Cyan (BGR)
        color_neg=(255, 0, 255)  # Magenta (BGR)
    )

    return new_frame


def run_eval(args):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"eval-{os.path.basename(args.checkpoint)}",
        config=vars(args),
    )

    print(f"Initializing {args.env_name}...")
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(args.env_name, auto_reset=False)
    env_params = env.default_params

    rng = jax.random.PRNGKey(args.seed)
    model = load_model(args.checkpoint, args.device)

    total_rewards = []
    
    print(f"Starting evaluation for {args.num_episodes} episodes...")

    for ep in range(args.num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        done = False
        ep_reward = 0
        step_count = 0

        # --- Buffers for Two-Pass Rendering ---
        raw_frames_buffer = [] # Store raw images here
        ep_values = []         # Store V(s)
        ep_rewards = []        # Store r
        
        while not done:
            # 1. Handle Observation Conversion
            obs_np = np.array(obs)

            # Convert to proper formats
            frame_raw = obs_to_255_range(obs_np)  # For video (uint8 0-255)
            obs_01 = obs_to_01_range(obs_np)      # For model (float 0-1)

            obs_tensor = torch.from_numpy(obs_01).float().to(args.device).unsqueeze(0)

            # 2. Model Forward
            with torch.no_grad():
                pi, v_tensor = model(obs_tensor)
                action_tensor = pi.sample()
                action = action_tensor.item()
                value_scalar = v_tensor.item()

            # 3. Buffer Data (Do NOT draw yet)
            ep_values.append(value_scalar)
            raw_frames_buffer.append(frame_raw)

            # 4. Step Env
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                step_key, env_state, action, env_params
            )

            ep_rewards.append(float(reward))
            ep_reward += float(reward)
            step_count += 1

            if done or step_count > 10000:
                break

        total_rewards.append(ep_reward)
        print(
            f"Episode {ep+1}: Reward = {ep_reward:.2f} (Steps: {step_count}, Mean V: {np.mean(ep_values):.3f})"
        )

        # --- POST-PROCESSING: Calculate Returns ---
        # Calculate Discounted Return to Go (G_t)
        gamma = 0.99
        returns_to_go = []
        G = 0
        for r in reversed(ep_rewards):
            G = r + gamma * G
            returns_to_go.insert(0, G)

        # --- POST-PROCESSING: Generate Video ---
        # Now we have both V(s) and G_t, we can draw the frames
        ep_frames_with_overlay = []
        
        # Determine loop length (handles rare off-by-one edge cases in buffers)
        loop_len = min(len(raw_frames_buffer), len(ep_values), len(returns_to_go))
        
        for i in range(loop_len):
            frame_overlay = draw_dual_value_bar(
                frame=raw_frames_buffer[i],
                pred_val=ep_values[i],
                true_val=returns_to_go[i],
                v_min=args.v_min,
                v_max=args.v_max
            )
            ep_frames_with_overlay.append(frame_overlay)

        # --- LOGGING: WandB ---
        # 1. Plot Line Series (Overlay V vs Return)
        steps_axis = [i for i in range(loop_len)]
        wandb.log(
            {
                f"eval/episode_{ep}_value_curve": wandb.plot.line_series(
                    xs=[steps_axis, steps_axis],
                    ys=[ep_values[:loop_len], returns_to_go[:loop_len]],
                    keys=["Predicted Value", "Empirical Return"],
                    title=f"Ep {ep} Value vs Return",
                    xname="step"
                )
            }
        )

        # 2. Upload Video
        if len(ep_frames_with_overlay) > 0:
            print("Processing video...")
            video_array = np.array(ep_frames_with_overlay)  # (T, H, W, C)
            video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
            wandb.log(
                {
                    "eval/trajectory_video": wandb.Video(
                        video_array, fps=15, format="mp4"
                    )
                }
            )
            print("Video uploaded to WandB.")

    avg_reward = np.mean(total_rewards)
    print(f"Evaluation Complete. Average Reward: {avg_reward:.2f}")

    wandb.log({"eval/avg_reward": avg_reward})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pth file"
    )
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--wandb_project", type=str, default="awr-rich-logs-v1")
    parser.add_argument("--wandb_entity", type=str, default="iris-sobolmark")
    # Visual config
    parser.add_argument(
        "--v_min", type=float, default=-1.0, help="Min value for video bar scaling"
    )
    parser.add_argument(
        "--v_max", type=float, default=10.0, help="Max value for video bar scaling"
    )

    run_eval(parser.parse_args())
