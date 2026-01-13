"""
Evaluation script for AWR augmented model using VLM server
"""
import os
import argparse
import torch
import numpy as np
import jax
import wandb
import cv2
import requests
import time
from pathlib import Path
from awr_aug import Config, ActorCriticConvAug

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_platform_name", "cpu")

def wait_for_server(server_url, timeout=60):
    """Wait for VLM server to be ready"""
    print(f"Waiting for VLM server at {server_url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"  Server ready! Model: {data['model']}")
                return True
        except:
            pass
        time.sleep(2)
    raise TimeoutError(f"VLM server at {server_url} not ready after {timeout}s")

def get_hidden_state_from_server(server_url, obs_np):
    """
    Get hidden state from VLM server

    Args:
        server_url: Base URL of VLM server (e.g., "http://babel-v5-16:5000")
        obs_np: Observation as numpy array (H, W, C) in 0-1 range

    Returns:
        numpy array of shape (2560,) - raw hidden state (not normalized)
    """
    # Send observation to server
    obs_list = obs_np.tolist()
    response = requests.post(
        f"{server_url}/get_hidden_state",
        json={'obs': obs_list},
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"Server error: {response.text}")

    data = response.json()
    hidden_state = np.array(data['hidden_state'], dtype=np.float32)
    return hidden_state

def load_model(checkpoint_path, device):
    """Load augmented policy model"""
    print(f"Loading checkpoint: {checkpoint_path}")
    model = ActorCriticConvAug(
        action_dim=Config.ACTION_DIM,
        layer_width=Config.LAYER_WIDTH,
        hidden_state_dim=Config.HIDDEN_STATE_DIM
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Policy loaded successfully!")
    return model

def load_normalization_stats(stats_path):
    """Load hidden state normalization statistics"""
    print(f"Loading normalization stats from: {stats_path}")
    data = np.load(stats_path)
    hidden_mean = data['mean']
    hidden_std = data['std']
    print(f"  Mean shape: {hidden_mean.shape}, range: [{hidden_mean.min():.3f}, {hidden_mean.max():.3f}]")
    print(f"  Std shape: {hidden_std.shape}, range: [{hidden_std.min():.3f}, {hidden_std.max():.3f}]")
    return hidden_mean, hidden_std

def draw_dual_line_graph(frame, values, rtgs, v_min=-1.0, v_max=10.0):
    """
    Draw dual line graph showing Value and Return-to-Go

    Args:
        frame: Image frame (H, W, C)
        values: List of value predictions
        rtgs: List of return-to-go values (if available, else None)
        v_min, v_max: Range for visualization

    Returns:
        Frame with graph overlay
    """
    frame = np.ascontiguousarray(frame)
    h, w, c = frame.shape

    # Add footer space
    footer_h = 60
    new_h = h + footer_h
    new_frame = np.zeros((new_h, w, c), dtype=frame.dtype)
    new_frame[0:h, 0:w] = frame

    if len(values) < 2:
        return new_frame

    # Graph dimensions
    graph_h = 40
    graph_w = w - 20
    x_start = 10
    y_start = h + 15

    # Draw background
    cv2.rectangle(new_frame, (x_start, y_start), (x_start + graph_w, y_start + graph_h), (30, 30, 30), -1)

    # Normalize values to graph coordinates
    def value_to_y(val):
        clamped = max(v_min, min(v_max, val))
        ratio = (clamped - v_min) / (v_max - v_min)
        return int(y_start + graph_h - ratio * graph_h)

    # Draw Value line (green)
    for i in range(len(values) - 1):
        x1 = int(x_start + (i / len(values)) * graph_w)
        x2 = int(x_start + ((i + 1) / len(values)) * graph_w)
        y1 = value_to_y(values[i])
        y2 = value_to_y(values[i + 1])
        cv2.line(new_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw RTG line (blue) if available
    if rtgs is not None and len(rtgs) > 1:
        for i in range(len(rtgs) - 1):
            x1 = int(x_start + (i / len(rtgs)) * graph_w)
            x2 = int(x_start + ((i + 1) / len(rtgs)) * graph_w)
            y1 = value_to_y(rtgs[i])
            y2 = value_to_y(rtgs[i + 1])
            cv2.line(new_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Add legend
    cv2.putText(new_frame, f"V: {values[-1]:.2f}", (x_start, h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    if rtgs is not None and len(rtgs) > 0:
        cv2.putText(new_frame, f"RTG: {rtgs[-1]:.2f}", (x_start + 80, h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return new_frame

def run_eval(args):
    """Run evaluation"""
    # Initialize WandB
    run_name = f"eval-aug-{os.path.basename(args.checkpoint).replace('.pth', '')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    print("[DEBUG] WandB initialized")

    # Wait for VLM server
    wait_for_server(args.server_url)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Load normalization stats
    hidden_mean, hidden_std = load_normalization_stats(args.stats_path)
    hidden_mean_tensor = torch.from_numpy(hidden_mean).float().to(device)
    hidden_std_tensor = torch.from_numpy(hidden_std).float().to(device)

    # Initialize environment
    print(f"Initializing {args.env_name}...")
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name(args.env_name, auto_reset=False)
    env_params = env.default_params

    rng = jax.random.PRNGKey(args.seed)

    # Run episodes
    all_returns = []
    all_lengths = []

    print(f"\n=== Episode 1/{args.num_episodes} ===")

    for ep in range(args.num_episodes):
        if ep > 0:
            print(f"\n=== Episode {ep+1}/{args.num_episodes} ===")

        # Reset environment
        print("[DEBUG] Resetting environment...")
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)
        print(f"[DEBUG] Environment reset. Obs shape: {obs.shape}")

        done = False
        ep_return = 0.0
        ep_length = 0
        ep_values = []
        ep_rewards = []
        ep_frames = []

        # Get first hidden state
        print("[DEBUG] Getting first hidden state from VLM server...")
        obs_np = np.array(obs)
        if obs_np.max() > 1.0:
            obs_np = obs_np / 255.0

        hidden_raw = get_hidden_state_from_server(args.server_url, obs_np)
        print(f"[DEBUG] Received hidden state. Shape: {hidden_raw.shape}")

        while not done and ep_length < 10000:
            # Normalize observation
            obs_np = np.array(obs)
            if obs_np.max() > 1.0:
                frame_raw = obs_np.astype(np.uint8)
                obs_np = obs_np / 255.0
            else:
                frame_raw = (obs_np * 255).astype(np.uint8)

            obs_tensor = torch.from_numpy(obs_np).float().to(device).unsqueeze(0)

            # Normalize hidden state
            hidden_tensor = torch.from_numpy(hidden_raw).float().to(device).unsqueeze(0)
            hidden_normalized = (hidden_tensor - hidden_mean_tensor) / hidden_std_tensor

            # Get action and value
            with torch.no_grad():
                pi, v_tensor = model(obs_tensor, hidden_normalized)
                action_tensor = pi.sample()
                action = action_tensor.item()
                value = v_tensor.item()

            if ep_length == 0:
                print(f"[DEBUG] Got action from policy: {action}")

            ep_values.append(value)

            # Step environment
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(step_key, env_state, action, env_params)

            ep_return += float(reward)
            ep_rewards.append(float(reward))
            ep_length += 1

            if ep_length == 1:
                print(f"[DEBUG] Environment stepped. Reward: {reward}, Done: {done}")

            # Save frame with visualization
            if args.save_video:
                # Compute RTG for visualization
                rtgs = [sum(ep_rewards[i:]) for i in range(len(ep_rewards))]
                frame_with_viz = draw_dual_line_graph(frame_raw.copy(), ep_values, rtgs, v_min=args.v_min, v_max=args.v_max)
                ep_frames.append(frame_with_viz)

            # Get next hidden state
            if not done:
                obs_np = np.array(obs)
                if obs_np.max() > 1.0:
                    obs_np = obs_np / 255.0
                hidden_raw = get_hidden_state_from_server(args.server_url, obs_np)

            # Progress updates
            if ep_length % 100 == 0:
                print(f"  Step {ep_length}: return={ep_return:.2f}")

        all_returns.append(ep_return)
        all_lengths.append(ep_length)

        print(f"Episode {ep+1} finished: return={ep_return:.2f}, length={ep_length}")

        # Save video
        if args.save_video and len(ep_frames) > 0:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{ep+1}.mp4"

            # Write video using cv2
            h, w = ep_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
            for frame in ep_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"  Video saved: {video_path}")

            # Log to WandB
            video_array = np.array(ep_frames)  # (T, H, W, C)
            video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
            wandb.log({f"eval/episode_{ep+1}_video": wandb.Video(video_array, fps=15, format="mp4")})

    # Summary
    print(f"\n=== Evaluation Results ===")
    print(f"Mean Return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"Mean Length: {np.mean(all_lengths):.2f}")
    print(f"All Returns: {all_returns}")

    wandb.log({
        "eval/mean_return": np.mean(all_returns),
        "eval/std_return": np.std(all_returns),
        "eval/mean_length": np.mean(all_lengths),
    })

    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--server_url', type=str, required=True, help='VLM server URL (e.g., http://babel-v5-16:5000)')
    parser.add_argument('--stats_path', type=str, required=True, help='Path to hidden_state_stats.npz')
    parser.add_argument('--env_name', type=str, default='Craftax-Pixels-v1')
    parser.add_argument('--num_episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_video', action='store_true', help='Save episode videos')
    parser.add_argument('--video_dir', type=str, default='./eval_videos')
    parser.add_argument('--wandb_project', type=str, default='craftax-offline-awr')
    parser.add_argument('--wandb_entity', type=str, default='iris-sobolmark')
    parser.add_argument('--v_min', type=float, default=-1.0)
    parser.add_argument('--v_max', type=float, default=10.0)

    args = parser.parse_args()
    run_eval(args)

if __name__ == "__main__":
    main()
