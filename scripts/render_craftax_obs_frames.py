#!/usr/bin/env python3
"""Render Craftax trajectory frames.

If `env_states/t_XXXXX.pbz2` exists, this script uses native
`render_craftax_pixels(state, ...)` for exact replay. Otherwise it decodes
symbolic `obs_vectors.npy` and reconstructs frames from Craftax textures.
"""

from __future__ import annotations

import argparse
import bz2
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from craftax.craftax.constants import BlockType, ItemType
from craftax.craftax.renderer import (
    INVENTORY_OBS_HEIGHT,
    OBS_DIM,
    TEXTURES,
    render_craftax_pixels,
)


NUM_BLOCK = len(BlockType)
NUM_ITEM = len(ItemType)
NUM_MOB_CLASSES = 5
NUM_MOB_TYPES_PER_CLASS = 8
NUM_MOB_CHANNELS = NUM_MOB_CLASSES * NUM_MOB_TYPES_PER_CLASS
MAP_CHANNELS = NUM_BLOCK + NUM_ITEM + NUM_MOB_CHANNELS + 1
MAP_LEN = OBS_DIM[0] * OBS_DIM[1] * MAP_CHANNELS

# Scalars after flattened map channels in render_craftax_symbolic.
INVENTORY_LEN = 16
POTIONS_LEN = 6
INTRINSICS_LEN = 9
DIRECTION_LEN = 4
ARMOUR_LEN = 4
ARMOUR_ENCHANTMENTS_LEN = 4
SPECIAL_LEN = 8

DIRECTION_OFFSET = MAP_LEN + INVENTORY_LEN + POTIONS_LEN + INTRINSICS_LEN
ARMOUR_OFFSET = DIRECTION_OFFSET + DIRECTION_LEN
ARMOUR_ENCHANTMENTS_OFFSET = ARMOUR_OFFSET + ARMOUR_LEN
SPECIAL_OFFSET = ARMOUR_ENCHANTMENTS_OFFSET + ARMOUR_ENCHANTMENTS_LEN


_ENV_STATE_RE = re.compile(r"^t_(\d{5})\.pbz2$")


def _load_env_state(path: Path):
    with bz2.BZ2File(path, "rb") as f:
        return pickle.load(f)


def _infer_total_steps_from_env_states(env_states_dir: Path) -> int:
    max_t = -1
    for p in env_states_dir.glob("t_*.pbz2"):
        m = _ENV_STATE_RE.match(p.name)
        if m is None:
            continue
        t = int(m.group(1))
        if t > max_t:
            max_t = t
    return max_t + 1 if max_t >= 0 else 0


def _overlay_rgba(base: np.ndarray, rgba: np.ndarray) -> None:
    """Alpha-composite rgba patch onto base (both HxWx3/4 uint8-like arrays)."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return
    alpha = (rgba[..., 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
    rgb = rgba[..., :3].astype(np.float32)
    base[...] = (base.astype(np.float32) * (1.0 - alpha) + rgb * alpha).clip(0.0, 255.0)


def _overlay_rgb_alpha(base: np.ndarray, rgb: np.ndarray, alpha_rgb: np.ndarray) -> None:
    """Alpha-composite rgb patch using alpha texture where alpha is encoded as RGB."""
    if rgb.ndim != 3 or alpha_rgb.ndim != 3:
        return
    # Craftax alpha textures are already normalized to [0, 1] for mobs/projectiles.
    # Keep backward compatibility with potential uint8 alpha by auto-scaling only when needed.
    alpha = alpha_rgb[..., :1].astype(np.float32)
    if float(alpha.max(initial=0.0)) > 1.0:
        alpha = alpha / 255.0
    alpha = alpha.clip(0.0, 1.0)
    base[...] = (base.astype(np.float32) * (1.0 - alpha) + rgb.astype(np.float32) * alpha).clip(0.0, 255.0)


def _tile_slice(row: int, col: int, bs: int) -> Tuple[slice, slice]:
    return slice(row * bs, (row + 1) * bs), slice(col * bs, (col + 1) * bs)


def _safe_slice(vec: np.ndarray, offset: int, length: int) -> np.ndarray:
    out = np.zeros((length,), dtype=np.float32)
    if offset >= vec.shape[0]:
        return out
    end = min(vec.shape[0], offset + length)
    out[: end - offset] = vec[offset:end]
    return out


def _decode_sqrt_scaled(v: float) -> int:
    return int(max(0, np.rint((max(float(v), 0.0) * 10.0) ** 2)))


def _decode_linear_scaled(v: float, scale: float) -> int:
    return int(np.rint(float(v) * float(scale)))


def _clip_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _decode_obs_scalars(vec: np.ndarray, textures: Dict[str, np.ndarray]) -> Dict[str, object]:
    inv = _safe_slice(vec, MAP_LEN, INVENTORY_LEN)
    potions = _safe_slice(vec, MAP_LEN + INVENTORY_LEN, POTIONS_LEN)
    intrinsics = _safe_slice(vec, MAP_LEN + INVENTORY_LEN + POTIONS_LEN, INTRINSICS_LEN)
    direction = _safe_slice(vec, DIRECTION_OFFSET, DIRECTION_LEN)
    armour = _safe_slice(vec, ARMOUR_OFFSET, ARMOUR_LEN)
    armour_ench = _safe_slice(vec, ARMOUR_ENCHANTMENTS_OFFSET, ARMOUR_ENCHANTMENTS_LEN)
    special = _safe_slice(vec, SPECIAL_OFFSET, SPECIAL_LEN)

    pickaxe_max = int(np.asarray(textures["pickaxe_textures"]).shape[0] - 1)
    sword_max = int(np.asarray(textures["sword_textures"]).shape[0] - 1)
    bow_max = int(np.asarray(textures["bow_textures"]).shape[0] - 1)
    armour_max = int(np.asarray(textures["armour_textures"]).shape[0] - 1)
    sword_ench_max = int(np.asarray(textures["sword_enchantment_textures"]).shape[0] - 1)
    arrow_ench_max = int(np.asarray(textures["arrow_enchantment_textures"]).shape[0] - 1)
    armour_ench_max = int(np.asarray(textures["armour_enchantment_textures"]).shape[0] - 1)

    direction_idx = int(np.argmax(direction)) if direction.size == DIRECTION_LEN else 0
    direction_idx = _clip_int(direction_idx, 0, 3)

    decoded = {
        "wood": _decode_sqrt_scaled(inv[0]),
        "stone": _decode_sqrt_scaled(inv[1]),
        "coal": _decode_sqrt_scaled(inv[2]),
        "iron": _decode_sqrt_scaled(inv[3]),
        "diamond": _decode_sqrt_scaled(inv[4]),
        "sapphire": _decode_sqrt_scaled(inv[5]),
        "ruby": _decode_sqrt_scaled(inv[6]),
        "sapling": _decode_sqrt_scaled(inv[7]),
        "torches": _decode_sqrt_scaled(inv[8]),
        "arrows": _decode_sqrt_scaled(inv[9]),
        "books": _clip_int(_decode_linear_scaled(inv[10], 2.0), 0, 99),
        "pickaxe": _clip_int(_decode_linear_scaled(inv[11], 4.0), 0, pickaxe_max),
        "sword": _clip_int(_decode_linear_scaled(inv[12], 4.0), 0, sword_max),
        "sword_enchantment": _clip_int(_decode_linear_scaled(inv[13], 1.0), 0, sword_ench_max),
        "bow_enchantment": _clip_int(_decode_linear_scaled(inv[14], 1.0), 0, arrow_ench_max),
        "bow": _clip_int(_decode_linear_scaled(inv[15], 1.0), 0, bow_max),
        "potions": [_decode_sqrt_scaled(v) for v in potions],
        "player_health": max(1, _decode_linear_scaled(intrinsics[0], 10.0)),
        "player_food": max(0, _decode_linear_scaled(intrinsics[1], 10.0)),
        "player_drink": max(0, _decode_linear_scaled(intrinsics[2], 10.0)),
        "player_energy": max(0, _decode_linear_scaled(intrinsics[3], 10.0)),
        "player_mana": max(0, _decode_linear_scaled(intrinsics[4], 10.0)),
        "player_xp": _clip_int(_decode_linear_scaled(intrinsics[5], 10.0), 0, 9),
        "player_dexterity": _clip_int(_decode_linear_scaled(intrinsics[6], 10.0), 0, 9),
        "player_strength": _clip_int(_decode_linear_scaled(intrinsics[7], 10.0), 0, 9),
        "player_intelligence": _clip_int(_decode_linear_scaled(intrinsics[8], 10.0), 0, 9),
        "direction_idx": direction_idx,
        "armour": [_clip_int(_decode_linear_scaled(v, 2.0), 0, armour_max) for v in armour],
        "armour_enchantments": [
            _clip_int(_decode_linear_scaled(v, 1.0), 0, armour_ench_max) for v in armour_ench
        ],
        "light_level": float(np.clip(special[0], 0.0, 1.0)),
        "is_sleeping": bool(special[1] > 0.5),
        "is_resting": bool(special[2] > 0.5),
        "learned_fireball": bool(special[3] > 0.5),
        "learned_iceball": bool(special[4] > 0.5),
        "player_level": _clip_int(_decode_linear_scaled(special[5], 10.0), 0, 9),
    }
    return decoded


def _render_inventory_panel(
    decoded: Dict[str, object],
    *,
    bs: int,
    textures: Dict[str, np.ndarray],
) -> np.ndarray:
    inv_pixel_left_space = (bs - int(0.8 * bs)) // 2
    inv_pixel_right_space = bs - int(0.8 * bs) - inv_pixel_left_space

    inv_pixels = np.zeros(
        (INVENTORY_OBS_HEIGHT * bs, OBS_DIM[1] * bs, 3),
        dtype=np.float32,
    )

    number_size = int(bs * 0.4)
    number_offset = bs - number_size
    number_double_offset = bs - 2 * number_size

    number_textures = np.asarray(textures["number_textures"], dtype=np.float32)
    number_textures_alpha = np.asarray(textures["number_textures_alpha"], dtype=np.float32)
    number_textures_with_zero = np.asarray(textures["number_textures_with_zero"], dtype=np.float32)
    number_textures_alpha_with_zero = np.asarray(
        textures["number_textures_alpha_with_zero"], dtype=np.float32
    )

    def _render_digit(number: int, x: int, y: int) -> None:
        number = _clip_int(number, 0, 9)
        tex = number_textures[number]
        alpha = number_textures_alpha[number]
        h, w = tex.shape[:2]
        r0 = y * bs + number_offset
        c0 = x * bs + number_offset
        r1 = min(inv_pixels.shape[0], r0 + h)
        c1 = min(inv_pixels.shape[1], c0 + w)
        tex = tex[: r1 - r0, : c1 - c0]
        alpha = alpha[: r1 - r0, : c1 - c0]
        inv_pixels[r0:r1, c0:c1] = inv_pixels[r0:r1, c0:c1] * (1.0 - alpha) + tex

    def _render_two_digit(number: int, x: int, y: int) -> None:
        number = _clip_int(number, 0, 99)
        tens = _clip_int(number // 10, 0, 9)
        ones = _clip_int(number % 10, 0, 9)
        if number == 0:
            ones_tex = number_textures[ones]
            ones_alpha = number_textures_alpha[ones]
        else:
            ones_tex = number_textures_with_zero[ones]
            ones_alpha = number_textures_alpha_with_zero[ones]

        h, w = ones_tex.shape[:2]
        r0 = y * bs + number_offset
        c0 = x * bs + number_offset
        r1 = min(inv_pixels.shape[0], r0 + h)
        c1 = min(inv_pixels.shape[1], c0 + w)
        t1 = ones_tex[: r1 - r0, : c1 - c0]
        a1 = ones_alpha[: r1 - r0, : c1 - c0]
        inv_pixels[r0:r1, c0:c1] = inv_pixels[r0:r1, c0:c1] * (1.0 - a1) + t1

        tens_tex = number_textures[tens]
        tens_alpha = number_textures_alpha[tens]
        h2, w2 = tens_tex.shape[:2]
        c0b = x * bs + number_double_offset
        c1b = min(inv_pixels.shape[1], c0b + w2)
        t2 = tens_tex[: r1 - r0, : c1b - c0b]
        a2 = tens_alpha[: r1 - r0, : c1b - c0b]
        inv_pixels[r0:r1, c0b:c1b] = inv_pixels[r0:r1, c0b:c1b] * (1.0 - a2) + t2

    def _render_icon(texture: np.ndarray, x: int, y: int) -> None:
        tex = np.asarray(texture, dtype=np.float32)
        r0 = bs * y + inv_pixel_left_space
        r1 = bs * (y + 1) - inv_pixel_right_space
        c0 = bs * x + inv_pixel_left_space
        c1 = bs * (x + 1) - inv_pixel_right_space
        h = min(r1 - r0, tex.shape[0])
        w = min(c1 - c0, tex.shape[1])
        inv_pixels[r0 : r0 + h, c0 : c0 + w] = tex[:h, :w, :3]

    def _render_icon_with_alpha(texture: np.ndarray, x: int, y: int) -> None:
        tex = np.asarray(texture, dtype=np.float32)
        r0 = bs * y + inv_pixel_left_space
        r1 = bs * (y + 1) - inv_pixel_right_space
        c0 = bs * x + inv_pixel_left_space
        c1 = bs * (x + 1) - inv_pixel_right_space
        h = min(r1 - r0, tex.shape[0])
        w = min(c1 - c0, tex.shape[1])
        base = inv_pixels[r0 : r0 + h, c0 : c0 + w]
        alpha = tex[:h, :w, 3:4]
        inv_pixels[r0 : r0 + h, c0 : c0 + w] = base * (1.0 - alpha) + tex[:h, :w, :3] * alpha

    empty = np.asarray(textures["smaller_empty_texture"], dtype=np.float32)
    smaller_blocks = np.asarray(textures["smaller_block_textures"], dtype=np.float32)

    # Stats
    health_tex = np.asarray(
        textures["health_texture"] if int(decoded["player_health"]) > 0 else empty,
        dtype=np.float32,
    )
    hunger_tex = np.asarray(
        textures["hunger_texture"] if int(decoded["player_food"]) > 0 else empty,
        dtype=np.float32,
    )
    thirst_tex = np.asarray(
        textures["thirst_texture"] if int(decoded["player_drink"]) > 0 else empty,
        dtype=np.float32,
    )
    energy_tex = np.asarray(
        textures["energy_texture"] if int(decoded["player_energy"]) > 0 else empty,
        dtype=np.float32,
    )
    mana_tex = np.asarray(
        textures["mana_texture"] if int(decoded["player_mana"]) > 0 else empty,
        dtype=np.float32,
    )
    _render_icon(health_tex, 0, 0)
    _render_two_digit(int(decoded["player_health"]), 0, 0)
    _render_icon(hunger_tex, 1, 0)
    _render_two_digit(int(decoded["player_food"]), 1, 0)
    _render_icon(thirst_tex, 2, 0)
    _render_two_digit(int(decoded["player_drink"]), 2, 0)
    _render_icon(energy_tex, 3, 0)
    _render_two_digit(int(decoded["player_energy"]), 3, 0)
    _render_icon(mana_tex, 4, 0)
    _render_two_digit(int(decoded["player_mana"]), 4, 0)

    # Resources and tools
    _render_icon(
        smaller_blocks[int(BlockType.WOOD.value)] if int(decoded["wood"]) > 0 else empty,
        0,
        2,
    )
    _render_two_digit(int(decoded["wood"]), 0, 2)
    _render_icon(
        smaller_blocks[int(BlockType.STONE.value)] if int(decoded["stone"]) > 0 else empty,
        1,
        2,
    )
    _render_two_digit(int(decoded["stone"]), 1, 2)
    _render_icon(
        smaller_blocks[int(BlockType.COAL.value)] if int(decoded["coal"]) > 0 else empty,
        0,
        1,
    )
    _render_two_digit(int(decoded["coal"]), 0, 1)
    _render_icon(
        smaller_blocks[int(BlockType.IRON.value)] if int(decoded["iron"]) > 0 else empty,
        1,
        1,
    )
    _render_two_digit(int(decoded["iron"]), 1, 1)
    _render_icon(
        smaller_blocks[int(BlockType.DIAMOND.value)] if int(decoded["diamond"]) > 0 else empty,
        2,
        1,
    )
    _render_two_digit(int(decoded["diamond"]), 2, 1)
    _render_icon(
        smaller_blocks[int(BlockType.SAPPHIRE.value)] if int(decoded["sapphire"]) > 0 else empty,
        3,
        1,
    )
    _render_two_digit(int(decoded["sapphire"]), 3, 1)
    _render_icon(
        smaller_blocks[int(BlockType.RUBY.value)] if int(decoded["ruby"]) > 0 else empty,
        4,
        1,
    )
    _render_two_digit(int(decoded["ruby"]), 4, 1)
    _render_icon(
        np.asarray(textures["sapling_texture"], dtype=np.float32) if int(decoded["sapling"]) > 0 else empty,
        5,
        1,
    )
    _render_two_digit(int(decoded["sapling"]), 5, 1)

    _render_icon(np.asarray(textures["pickaxe_textures"], dtype=np.float32)[int(decoded["pickaxe"])], 8, 2)
    _render_icon(np.asarray(textures["sword_textures"], dtype=np.float32)[int(decoded["sword"])], 8, 1)
    _render_icon(np.asarray(textures["bow_textures"], dtype=np.float32)[int(decoded["bow"])], 6, 1)
    _render_icon(
        np.asarray(textures["player_projectile_textures"], dtype=np.float32)[0]
        if int(decoded["arrows"]) > 0
        else empty,
        6,
        2,
    )
    _render_two_digit(int(decoded["arrows"]), 6, 2)

    armour_tex = np.asarray(textures["armour_textures"], dtype=np.float32)
    for i in range(4):
        _render_icon(armour_tex[int(decoded["armour"][i]), i], 7, i)

    _render_icon(
        np.asarray(textures["torch_inv_texture"], dtype=np.float32) if int(decoded["torches"]) > 0 else empty,
        2,
        2,
    )
    _render_two_digit(int(decoded["torches"]), 2, 2)

    potion_tex = np.asarray(textures["potion_textures"], dtype=np.float32)
    for p_idx in range(min(POTIONS_LEN, len(decoded["potions"]))):
        _render_icon(potion_tex[p_idx] if int(decoded["potions"][p_idx]) > 0 else empty, p_idx, 3)
        _render_two_digit(int(decoded["potions"][p_idx]), p_idx, 3)

    _render_icon(
        np.asarray(textures["book_texture"], dtype=np.float32) if int(decoded["books"]) > 0 else empty,
        3,
        2,
    )
    _render_two_digit(int(decoded["books"]), 3, 2)

    _render_icon(
        np.asarray(textures["fireball_inv_texture"], dtype=np.float32)
        if bool(decoded["learned_fireball"])
        else empty,
        4,
        2,
    )
    _render_icon(
        np.asarray(textures["iceball_inv_texture"], dtype=np.float32)
        if bool(decoded["learned_iceball"])
        else empty,
        5,
        2,
    )

    sword_ench_tex = np.asarray(textures["sword_enchantment_textures"], dtype=np.float32)
    arrow_ench_tex = np.asarray(textures["arrow_enchantment_textures"], dtype=np.float32)
    armour_ench_tex = np.asarray(textures["armour_enchantment_textures"], dtype=np.float32)
    _render_icon_with_alpha(sword_ench_tex[int(decoded["sword_enchantment"])], 8, 1)
    arrow_ench_level = int(decoded["bow_enchantment"]) if int(decoded["arrows"]) > 0 else 0
    _render_icon_with_alpha(arrow_ench_tex[arrow_ench_level], 6, 2)
    for i in range(4):
        _render_icon_with_alpha(armour_ench_tex[int(decoded["armour_enchantments"][i]), i], 7, i)

    # Level + attributes
    _render_digit(int(decoded["player_level"]), 6, 0)
    _render_icon(
        np.asarray(textures["xp_texture"], dtype=np.float32) if int(decoded["player_xp"]) > 0 else empty,
        9,
        0,
    )
    _render_digit(int(decoded["player_xp"]), 9, 0)
    _render_icon(np.asarray(textures["dex_texture"], dtype=np.float32), 9, 1)
    _render_digit(int(decoded["player_dexterity"]), 9, 1)
    _render_icon(np.asarray(textures["str_texture"], dtype=np.float32), 9, 2)
    _render_digit(int(decoded["player_strength"]), 9, 2)
    _render_icon(np.asarray(textures["int_texture"], dtype=np.float32), 9, 3)
    _render_digit(int(decoded["player_intelligence"]), 9, 3)

    return inv_pixels.clip(0.0, 255.0).astype(np.uint8)


def _decode_and_render_frame(
    vec: np.ndarray,
    *,
    bs: int,
    textures: Dict[str, np.ndarray],
) -> np.ndarray:
    decoded = _decode_obs_scalars(vec, textures)
    h, w = OBS_DIM
    all_map = vec[:MAP_LEN].reshape(h, w, MAP_CHANNELS)

    block_logits = all_map[..., :NUM_BLOCK]
    item_logits = all_map[..., NUM_BLOCK : NUM_BLOCK + NUM_ITEM]
    mob_logits = all_map[
        ...,
        NUM_BLOCK + NUM_ITEM : NUM_BLOCK + NUM_ITEM + NUM_MOB_CHANNELS,
    ].reshape(h, w, NUM_MOB_CLASSES, NUM_MOB_TYPES_PER_CLASS)
    light_mask = all_map[..., -1] > 0.5

    block_ids = np.argmax(block_logits, axis=-1).astype(np.int32)
    # If a cell is dark, symbolic channels are masked. Force darkness tile.
    dark_id = int(BlockType.DARKNESS.value)
    block_ids = np.where(light_mask, block_ids, dark_id)

    block_tex = textures["block_textures"][block_ids]  # [H, W, bs, bs, 3]
    frame = (
        block_tex.transpose(0, 2, 1, 3, 4).reshape(h * bs, w * bs, 3).astype(np.float32)
    )

    item_ids = np.argmax(item_logits, axis=-1).astype(np.int32)
    item_full = textures["full_map_item_textures"]  # [num_item, H*bs, W*bs, 4]
    for r in range(h):
        for c in range(w):
            if not light_mask[r, c]:
                continue
            item_id = int(item_ids[r, c])
            if item_id == int(ItemType.NONE.value):
                continue
            rs, cs = _tile_slice(r, c, bs)
            patch = item_full[item_id, rs, cs, :]
            _overlay_rgba(frame[rs, cs, :], patch)

    melee_tex = textures["melee_mob_textures"]
    melee_alpha = textures["melee_mob_texture_alphas"]
    passive_tex = textures["passive_mob_textures"]
    passive_alpha = textures["passive_mob_texture_alphas"]
    ranged_tex = textures["ranged_mob_textures"]
    ranged_alpha = textures["ranged_mob_texture_alphas"]
    proj_tex = textures["projectile_textures"]
    proj_alpha = textures["projectile_texture_alphas"]
    pproj_tex = textures["player_projectile_textures"]

    for r in range(h):
        for c in range(w):
            if not light_mask[r, c]:
                continue
            rs, cs = _tile_slice(r, c, bs)
            # Overlay order: mobs then projectiles, so projectiles are most visible.
            for cls_idx, type_idx in zip(*np.where(mob_logits[r, c] > 0.5)):
                cls_idx = int(cls_idx)
                type_idx = int(type_idx)
                if cls_idx == 0 and type_idx < melee_tex.shape[0]:
                    _overlay_rgb_alpha(frame[rs, cs, :], melee_tex[type_idx], melee_alpha[type_idx])
                elif cls_idx == 1 and type_idx < passive_tex.shape[0]:
                    _overlay_rgb_alpha(frame[rs, cs, :], passive_tex[type_idx], passive_alpha[type_idx])
                elif cls_idx == 2 and type_idx < ranged_tex.shape[0]:
                    _overlay_rgb_alpha(frame[rs, cs, :], ranged_tex[type_idx], ranged_alpha[type_idx])
                elif cls_idx == 3 and type_idx < proj_tex.shape[0]:
                    _overlay_rgb_alpha(frame[rs, cs, :], proj_tex[type_idx], proj_alpha[type_idx])
                elif cls_idx == 4 and type_idx < pproj_tex.shape[0]:
                    patch = pproj_tex[type_idx]
                    alpha = (np.sum(patch, axis=-1, keepdims=True) > 0).astype(np.float32)
                    frame[rs, cs, :] = (
                        frame[rs, cs, :].astype(np.float32) * (1.0 - alpha)
                        + patch.astype(np.float32) * alpha
                    ).clip(0.0, 255.0)

    player_texture_index = 4 if bool(decoded["is_sleeping"]) else int(decoded["direction_idx"])
    player_texture_index = _clip_int(player_texture_index, 0, 4)
    player_rgb = np.asarray(textures["full_map_player_textures"], dtype=np.float32)[player_texture_index]
    player_alpha = np.asarray(textures["full_map_player_textures_alpha"], dtype=np.float32)[player_texture_index]
    frame = frame * (1.0 - player_alpha) + player_rgb * player_alpha

    frame_uint8 = frame.clip(0.0, 255.0).astype(np.uint8)
    inv_panel = _render_inventory_panel(decoded, bs=bs, textures=textures)
    return np.concatenate([frame_uint8, inv_panel], axis=0).astype(np.uint8)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trajectory-dir",
        type=Path,
        required=True,
        help="Trajectory dir containing obs_vectors.npy and/or env_states/",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Frame output dir (default: <trajectory-dir>/render_frames_bs<block-size>)")
    p.add_argument("--block-size", type=int, default=16, choices=[10, 16, 64], help="Craftax block pixel size")
    p.add_argument("--stride", type=int, default=1, help="Render every Nth timestep")
    p.add_argument("--start-t", type=int, default=0, help="Start timestep index")
    p.add_argument("--end-t", type=int, default=-1, help="Inclusive end timestep index; -1 means last")
    p.add_argument("--max-frames", type=int, default=0, help="If >0, cap number of rendered frames")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing frame files")
    p.add_argument(
        "--prefer-env-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If env_states/t_XXXXX.pbz2 exists, use render_craftax_pixels(state, ...) "
            "for exact rendering; otherwise fall back to obs-vector decoding."
        ),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    trajectory_dir = args.trajectory_dir.resolve()
    obs_path = trajectory_dir / "obs_vectors.npy"
    env_states_dir = trajectory_dir / "env_states"
    has_env_states = env_states_dir.exists()
    use_env_states = bool(args.prefer_env_state and has_env_states)

    bs = int(args.block_size)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (trajectory_dir / f"render_frames_bs{bs}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    obs: Optional[np.ndarray] = None
    tx: Optional[Dict[str, np.ndarray]] = None
    if obs_path.exists():
        obs = np.load(obs_path)
        if obs.ndim != 2:
            raise ValueError(f"Expected [T, D] obs array, got shape {obs.shape}")
        if obs.shape[1] < SPECIAL_OFFSET + SPECIAL_LEN:
            raise ValueError(f"Obs dim too small ({obs.shape[1]}) for expected symbolic layout.")
        tx = {k: np.asarray(v) for k, v in TEXTURES[bs].items()}

    if obs is None and not use_env_states:
        raise FileNotFoundError(
            f"Missing obs_vectors.npy: {obs_path} (and no env_states dir for exact render fallback)."
        )

    if obs is not None:
        total_t = int(obs.shape[0])
    else:
        total_t = _infer_total_steps_from_env_states(env_states_dir)
    if total_t <= 0:
        raise ValueError("Could not infer trajectory length from obs vectors or env_states snapshots.")

    start_t = max(0, int(args.start_t))
    end_t = total_t - 1 if int(args.end_t) < 0 else min(total_t - 1, int(args.end_t))
    stride = max(1, int(args.stride))

    ts = list(range(start_t, end_t + 1, stride))
    if args.max_frames > 0:
        ts = ts[: int(args.max_frames)]

    rendered = 0
    skipped_existing = 0
    rendered_from_env_state = 0
    rendered_from_obs_decode = 0
    missing_state_snapshots = 0
    for i, t in enumerate(ts, 1):
        out_path = output_dir / f"t_{t:05d}.png"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        frame: Optional[np.ndarray] = None
        if use_env_states:
            state_path = env_states_dir / f"t_{t:05d}.pbz2"
            if state_path.exists():
                state = _load_env_state(state_path)
                frame = np.asarray(
                    render_craftax_pixels(state, bs, do_night_noise=False),
                    dtype=np.uint8,
                )
                rendered_from_env_state += 1
            else:
                missing_state_snapshots += 1
        if frame is None:
            if obs is None or tx is None:
                continue
            frame = _decode_and_render_frame(obs[t], bs=bs, textures=tx)
            rendered_from_obs_decode += 1
        Image.fromarray(frame).save(out_path)
        rendered += 1
        if i % 100 == 0 or i == len(ts):
            print(f"progress: {i}/{len(ts)}")

    meta = {
        "trajectory_dir": str(trajectory_dir),
        "obs_vectors_path": (str(obs_path) if obs_path.exists() else ""),
        "env_states_dir": (str(env_states_dir) if has_env_states else ""),
        "prefer_env_state": bool(args.prefer_env_state),
        "used_env_state_rendering": bool(use_env_states),
        "output_dir": str(output_dir),
        "block_size": bs,
        "obs_shape": ([int(obs.shape[0]), int(obs.shape[1])] if obs is not None else []),
        "frame_shape": [int((OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * bs), int(OBS_DIM[1] * bs), 3],
        "start_t": start_t,
        "end_t": end_t,
        "stride": stride,
        "requested_frames": len(ts),
        "rendered_frames": rendered,
        "rendered_from_env_state": rendered_from_env_state,
        "rendered_from_obs_decode": rendered_from_obs_decode,
        "missing_state_snapshots": missing_state_snapshots,
        "skipped_existing": skipped_existing,
        "filename_pattern": "t_00000.png",
    }
    (output_dir / "frames_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"output_dir: {output_dir}")
    print(f"requested_frames: {len(ts)}")
    print(f"rendered_frames: {rendered}")
    print(f"rendered_from_env_state: {rendered_from_env_state}")
    print(f"rendered_from_obs_decode: {rendered_from_obs_decode}")
    print(f"missing_state_snapshots: {missing_state_snapshots}")
    print(f"skipped_existing: {skipped_existing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
