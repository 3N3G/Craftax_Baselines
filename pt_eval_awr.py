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


def draw_value_bar(frame, value, v_min=-1.0, v_max=10.0):
    """
    Draws a health-bar style indicator for the Value function
    as pixels BELOW the current scene (extending the frame height).
    """
    # Ensure frame is contiguous numpy array for cv2
    frame = np.ascontiguousarray(frame)
    h, w, c = frame.shape

    # Define footer height (pixels to add below)
    footer_h = 30
    new_h = h + footer_h

    # Create new frame with extra height (initialized to black/dark background)
    new_frame = np.zeros((new_h, w, c), dtype=frame.dtype)

    # Copy the original scene onto the top of the new frame
    new_frame[0:h, 0:w] = frame

    # Bar Configuration
    bar_h = 10
    bar_w = w - 20
    x_start = 10

    # Position elements in the new footer area (below original h)
    y_start = h + 15  # Bar top
    text_y = h + 10  # Text baseline

    # Normalize value for bar width
    # Clamp value between min and max for visualization
    clamped_val = max(v_min, min(v_max, value))
    ratio = (clamped_val - v_min) / (v_max - v_min)
    fill_w = int(ratio * bar_w)

    # Colors (BGR for OpenCV)
    bg_color = (50, 50, 50)
    fill_color = (0, 255, 0) if value > 0 else (0, 0, 255)
    text_color = (255, 255, 255)

    # Draw Background for the bar
    cv2.rectangle(
        new_frame, (x_start, y_start), (x_start + bar_w, y_start + bar_h), bg_color, -1
    )
    # Draw Fill
    if fill_w > 0:
        cv2.rectangle(
            new_frame,
            (x_start, y_start),
            (x_start + fill_w, y_start + bar_h),
            fill_color,
            -1,
        )

    # Draw Text Value
    text = f"V: {value:.3f}"
    cv2.putText(
        new_frame, text, (x_start, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1
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
    video_frames = []

    print(f"Starting evaluation for {args.num_episodes} episodes...")

    for ep in range(args.num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        done = False
        ep_reward = 0
        ep_frames = []
        ep_values = []  # Store V(s) for this episode
        step_count = 0

        while not done:
            # 1. Handle Observation Conversion
            obs_np = np.array(obs)

            # Convert to proper formats using helpers
            frame_raw = obs_to_255_range(obs_np)  # For video (uint8 0-255)
            obs_01 = obs_to_01_range(obs_np)  # For model (float 0-1)

            obs_tensor = torch.from_numpy(obs_01).float().to(args.device).unsqueeze(0)

            # 2. Model Forward
            with torch.no_grad():
                pi, v_tensor = model(obs_tensor)
                action_tensor = pi.sample()
                action = action_tensor.item()
                value_scalar = v_tensor.item()

            # Track V
            ep_values.append(value_scalar)

            # 3. Draw Video Frame
            # if ep == 0:
            # Draw the Value Bar on the frame
            frame_with_overlay = draw_value_bar(
                frame_raw.copy(), value_scalar, v_min=args.v_min, v_max=args.v_max
            )
            ep_frames.append(frame_with_overlay)

            # 4. Step Env
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                step_key, env_state, action, env_params
            )

            ep_reward += float(reward)
            step_count += 1

            if done or step_count > 10000:
                break

        total_rewards.append(ep_reward)
        print(
            f"Episode {ep+1}: Reward = {ep_reward:.2f} (Steps: {step_count}, Mean V: {np.mean(ep_values):.3f})"
        )

        # LOGGING: Average V over time for this trajectory
        # Create a custom x-axis plot for this episode
        data = [[x, y] for (x, y) in zip(range(len(ep_values)), ep_values)]
        table = wandb.Table(data=data, columns=["step", "value"])
        wandb.log(
            {
                f"eval/episode_{ep}_value_curve": wandb.plot.line(
                    table, "step", "value", title=f"Ep {ep} Value over Time"
                )
            }
        )

        if len(ep_frames) > 0:
            print("Processing video...")
            video_array = np.array(ep_frames)  # (T, H, W, C)
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
