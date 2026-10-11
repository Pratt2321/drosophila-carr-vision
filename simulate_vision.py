#!/usr/bin/env python3
"""
Drosophila melanogaster Embodied 3D Simulation & Compound-Eye Vision Pipeline
=============================================================================
Simulates an embodied Drosophila melanogaster in FlyGym's MuJoCo 3D arena
reacting to basketball broadcast highlights (Coen Carr fast break).

Features:
  1. 3D Embodied Simulation:
     - FlyGym NeuroMechFly in FlatGroundWorld with active leg joints.
     - Third-person over-the-shoulder 3D camera tracking the fly.
     - Embodied Giant Fiber escape reflex animating leg extension/jumping.
  2. Biological Retinal Mapping:
     - Maps video frames to 721 ommatidia per eye using FlyGym's Retina.
     - Simulates pale (green) and yellow (UV) photoreceptor spectral channels.
  3. Looming & Giant Fiber Circuit:
     - Temporal derivative ΔI(t) on central receptive field.
     - Triggers stereotyped escape reflex (IDLE -> FREEZE -> JUMP -> FLIGHT).
  4. Real-time Compositing & HUD:
     - Main viewport: 3D fly body reacting in MuJoCo arena.
     - Top-Right Inset: Virtual screen displaying the broadcast video.
     - Bottom-Right Inset: Reconstructed hexagonal ommatidia vision.
     - Bottom-Left HUD: Giant Fiber state badge, looming bar, timecode.
  5. H.264 Video Encoding:
     - Web- and VS Code-compatible MP4 (libx264, yuv420p, +faststart).

Usage:
    python simulate_vision.py --start-sec 140 --count 600
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# FlyGym & MuJoCo imports with graceful handling
# ---------------------------------------------------------------------------

_USE_FLYGYM = False
try:
    import mujoco
    from flygym.anatomy import AxisOrder, JointPreset, Skeleton
    from flygym.compose.fly.neuromechfly import NeuroMechFly
    from flygym.compose.pose import KinematicPosePreset
    from flygym.compose.world.flat_ground import FlatGroundWorld
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from flygym.vision.retina import Retina as _FlyGymRetina

    _USE_FLYGYM = True
    print("[✓] FlyGym and MuJoCo loaded successfully.")
except Exception as exc:
    print(f"[!] FlyGym/MuJoCo import failed ({exc}); running fallback mode.")


# ============================================================================
# Retinal Model
# ============================================================================

class HexRetina:
    """Hexagonal compound-eye retina model (~721 ommatidia per eye)."""

    NUM_OMMATIDIA = 721

    def __init__(self) -> None:
        self._flygym_retina = None

        if _USE_FLYGYM:
            try:
                self._flygym_retina = _FlyGymRetina()
                self.num_ommatidia = self._flygym_retina.num_ommatidia_per_eye
                self._retina_h = self._flygym_retina.nrows
                self._retina_w = self._flygym_retina.ncols
                print(
                    f"    FlyGym retina active: {self.num_ommatidia} ommatidia, "
                    f"res {self._retina_h}×{self._retina_w}"
                )
            except Exception as exc:
                print(f"[!] FlyGym Retina init failed ({exc}); falling back.")
                self._flygym_retina = None

        if self._flygym_retina is None:
            self.num_ommatidia = self.NUM_OMMATIDIA
            self._build_hex_grid()

    def _build_hex_grid(self) -> None:
        """Fallback hex grid model."""
        self._retina_h = 128
        self._retina_w = 128
        n_target = self.NUM_OMMATIDIA
        cols = int(np.ceil(np.sqrt(n_target * 1.15)))
        rows = int(np.ceil(n_target / cols))
        spacing_x = self._retina_w / (cols + 0.5)
        spacing_y = self._retina_h / rows

        centres = []
        for r in range(rows):
            for c in range(cols):
                cx = spacing_x * (c + 0.5 * (r % 2) + 0.25)
                cy = spacing_y * (r + 0.5)
                if cx < self._retina_w and cy < self._retina_h:
                    centres.append((cy, cx))
                if len(centres) >= n_target:
                    break
            if len(centres) >= n_target:
                break
        self._centres = np.array(centres[:n_target])
        self.num_ommatidia = len(self._centres)

        yy, xx = np.mgrid[0:self._retina_h, 0:self._retina_w]
        coords = np.stack([yy, xx], axis=-1).astype(np.float32)
        diff = coords[..., np.newaxis, :] - self._centres[np.newaxis, np.newaxis, :, :]
        dist2 = (diff ** 2).sum(axis=-1)
        self._omm_map = dist2.argmin(axis=-1)

        self._pale_mask = np.zeros(self.num_ommatidia, dtype=bool)
        self._pale_mask[::2] = True

    def raw_to_ommatidia(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Map BGR video frame to ommatidia intensities (N, 2)."""
        small = cv2.resize(
            frame_bgr, (self._retina_w, self._retina_h), interpolation=cv2.INTER_AREA
        )

        if self._flygym_retina is not None:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            return self._flygym_retina.raw_image_to_hex_pxls(rgb)

        green = small[:, :, 1].astype(np.float32) / 255.0
        blue = small[:, :, 0].astype(np.float32) / 255.0

        intensities = np.zeros((self.num_ommatidia, 2), dtype=np.float32)
        for i in range(self.num_ommatidia):
            mask = self._omm_map == i
            if mask.any():
                if self._pale_mask[i]:
                    intensities[i, 0] = green[mask].mean()
                else:
                    intensities[i, 1] = blue[mask].mean()
        return intensities

    def ommatidia_to_image(
        self, intensities: np.ndarray, out_h: int, out_w: int
    ) -> np.ndarray:
        """Reconstruct 2D biological hexagonal image from ommatidia intensities."""
        if self._flygym_retina is not None:
            human = self._flygym_retina.hex_pxls_to_human_readable(
                intensities, color_8bit=True
            )
            canvas = np.zeros((*human.shape[:2], 3), dtype=np.uint8)
            canvas[..., 1] = human[..., 0]                             # Pale (green)
            canvas[..., 0] = (human[..., 1] * 0.85).astype(np.uint8)   # Yellow (UV/blue)
            canvas[..., 2] = (human[..., 1] * 0.45).astype(np.uint8)   # Violet tint
            return cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        val_green = (intensities[:, 0] * 255).astype(np.uint8)[self._omm_map]
        val_uv = (intensities[:, 1] * 255).astype(np.uint8)[self._omm_map]
        canvas = np.zeros((self._retina_h, self._retina_w, 3), dtype=np.uint8)
        canvas[self._pale_mask[self._omm_map], 1] = val_green[self._pale_mask[self._omm_map]]
        canvas[~self._pale_mask[self._omm_map], 0] = (val_uv[~self._pale_mask[self._omm_map]] * 0.85).astype(np.uint8)
        canvas[~self._pale_mask[self._omm_map], 2] = (val_uv[~self._pale_mask[self._omm_map]] * 0.45).astype(np.uint8)
        return cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


# ============================================================================
# Looming / Optical-Flow Detector
# ============================================================================

class LoomingDetector:
    """Lobula plate / Giant Fiber looming pathway model."""

    def __init__(
        self,
        num_ommatidia: int,
        *,
        central_fraction: float = 0.35,
        base_threshold: float = 0.06,
        adaptation_tau: float = 0.90,
        cooldown_frames: int = 15,
    ) -> None:
        self.num_ommatidia = num_ommatidia
        lo = int(num_ommatidia * (0.5 - central_fraction / 2))
        hi = int(num_ommatidia * (0.5 + central_fraction / 2))
        self._central_ids = np.arange(lo, hi)

        self.base_threshold = base_threshold
        self.adaptation_tau = adaptation_tau
        self.cooldown_frames = cooldown_frames

        self._prev_intensities: np.ndarray | None = None
        self._adapted_threshold = base_threshold
        self._cooldown_remaining = 0

    def update(self, intensities: np.ndarray) -> tuple[float, bool]:
        combined = intensities.sum(axis=1)

        if self._prev_intensities is None:
            self._prev_intensities = combined.copy()
            return 0.0, False

        delta = combined - self._prev_intensities
        self._prev_intensities = combined.copy()

        # ------------------------------------------------------------------
        # Broadcast Cut Filter (Prevent False Looming Spikes)
        # ------------------------------------------------------------------
        # Hard camera angle cuts cause sudden whole-frame jumps across almost
        # all ommatidia simultaneously. Discard from looming accumulator if
        # whole-frame delta or jumping facet fraction exceeds cut threshold.
        global_abs = np.abs(delta)
        mean_global_delta = float(np.mean(global_abs))
        fraction_jump = float(np.mean(global_abs > 0.12))
        if fraction_jump > 0.50 or mean_global_delta > 0.19:
            # Broadcast cut detected: discard from looming accumulator
            return 0.0, False

        central_delta = delta[self._central_ids]
        expansion = float(np.mean(np.clip(central_delta, 0, None)))
        expansion += 0.5 * float(np.std(central_delta))

        self._adapted_threshold = (
            self.adaptation_tau * self._adapted_threshold
            + (1 - self.adaptation_tau) * expansion
        )
        effective_threshold = max(
            self.base_threshold, self._adapted_threshold * 1.8
        )

        triggered = False
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        elif expansion > effective_threshold:
            triggered = True
            self._cooldown_remaining = self.cooldown_frames

        return expansion, triggered


# ============================================================================
# Escape Reflex Kinematics State Machine
# ============================================================================

class EscapeReflexSimulator:
    """Giant Fiber stereotyped escape jump state machine."""

    def __init__(
        self,
        *,
        freeze_frames: int = 4,
        jump_frames: int = 6,
        flight_frames: int = 10,
        recovery_frames: int = 12,
    ) -> None:
        self.freeze_frames = freeze_frames
        self.jump_frames = jump_frames
        self.flight_frames = flight_frames
        self.recovery_frames = recovery_frames
        self._phase = "IDLE"
        self._phase_counter = 0
        self._total_triggers = 0

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def total_triggers(self) -> int:
        return self._total_triggers

    def trigger_escape(self) -> None:
        if self._phase == "IDLE":
            self._phase = "FREEZE"
            self._phase_counter = self.freeze_frames
            self._total_triggers += 1

    def step(self) -> str:
        if self._phase == "IDLE":
            return self._phase

        self._phase_counter -= 1
        if self._phase_counter <= 0:
            if self._phase == "FREEZE":
                self._phase = "JUMP"
                self._phase_counter = self.jump_frames
            elif self._phase == "JUMP":
                self._phase = "FLIGHT"
                self._phase_counter = self.flight_frames
            elif self._phase == "FLIGHT":
                self._phase = "RECOVERY"
                self._phase_counter = self.recovery_frames
            elif self._phase == "RECOVERY":
                self._phase = "IDLE"
                self._phase_counter = 0

        return self._phase


# ============================================================================
# Embodied 3D Fly Simulator (MuJoCo / FlyGym)
# ============================================================================

class Embodied3DFlySim:
    """Manages the 3D NeuroMechFly physics simulation and third-person camera."""

    def __init__(self, render_h: int = 540, render_w: int = 960) -> None:
        self.render_h = render_h
        self.render_w = render_w
        self._has_sim = False

        if not _USE_FLYGYM:
            return

        try:
            print("    Initializing NeuroMechFly in FlatGroundWorld...")
            fly = NeuroMechFly()
            skel = Skeleton(
                joint_preset=JointPreset.LEGS_ACTIVE_ONLY,
                axis_order=AxisOrder.ROLL_PITCH_YAW,
            )
            fly.add_joints(skel, neutral_pose=KinematicPosePreset.NEUTRAL)
            jointdofs = list(skel.iter_jointdofs(fly.root_segment))
            fly.add_actuators(
                jointdofs,
                actuator_type="position",
                neutral_input=KinematicPosePreset.NEUTRAL,
            )
            fly.add_vision()

            # Add articulated wing hinge joints for physical flight postures
            for b in fly.mjcf_root.bodies:
                if b.name in ("l_wing", "r_wing"):
                    j_yaw = b.add_joint()
                    j_yaw.name = f"{b.name}_yaw"
                    j_yaw.type = mujoco.mjtJoint.mjJNT_HINGE
                    j_yaw.axis = [0, 0, 1]
                    j_yaw.range = [-1.5, 1.5]

                    j_pitch = b.add_joint()
                    j_pitch.name = f"{b.name}_pitch"
                    j_pitch.type = mujoco.mjtJoint.mjJNT_HINGE
                    j_pitch.axis = [0, 1, 0]
                    j_pitch.range = [-1.5, 1.5]

            world = FlatGroundWorld()
            world.add_fly(
                fly,
                spawn_position=np.array([0.0, 0.0, 1.5]),
                spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
                add_ground_contact_sensors=False,
            )

            # Disable floor reflectance to eliminate ghost reflection clipping underneath floor
            for mat in world.mjcf_root.materials:
                if hasattr(mat, "reflectance"):
                    mat.reflectance = 0.0

            # Add third-person over-the-shoulder camera targeting the fly
            spec = world.mjcf_root
            spec.worldbody.add_camera(
                name="over_shoulder_cam",
                pos=[-6.86, -3.43, 6.23],
                xyaxes=[0.4067, -0.9135, 0.0, 0.4289, 0.191, 0.8829],
                fovy=46.0,
            )

            self.sim = Simulation(world)
            self.sim.set_renderer(
                "over_shoulder_cam", camera_res=(render_h, render_w)
            )
            self.sim.reset()

            self.cam_id = mujoco.mj_name2id(
                self.sim.mj_model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "over_shoulder_cam",
            )

            # Cache neutral qpos posture and ensure upright orientation [1, 0, 0, 0]
            self.neutral_qpos = self.sim.mj_data.qpos.copy()
            self.neutral_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.root_z_idx = 2

            # Identify leg and wing joint qpos addresses
            def get_joint_adr(name: str) -> int:
                jid = mujoco.mj_name2id(
                    self.sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                return self.sim.mj_model.jnt_qposadr[jid] if jid != -1 else -1

            self.wing_joints = {
                "l_yaw": get_joint_adr("nmf/l_wing_yaw"),
                "l_pitch": get_joint_adr("nmf/l_wing_pitch"),
                "r_yaw": get_joint_adr("nmf/r_wing_yaw"),
                "r_pitch": get_joint_adr("nmf/r_wing_pitch"),
            }

            self.leg_joints = {}
            for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
                self.leg_joints[f"{leg}_femur"] = get_joint_adr(
                    f"nmf/{leg}_coxa-{leg}_trochanterfemur-pitch"
                )
                self.leg_joints[f"{leg}_tibia"] = get_joint_adr(
                    f"nmf/{leg}_trochanterfemur-{leg}_tibia-pitch"
                )

            # State tracking for smooth recovery interpolation
            self.recovery_step = 0
            self.recovery_total = 12
            self.flight_snapshot_qpos: np.ndarray | None = None
            self.wing_flutter_phase = 0.0

            self._has_sim = True
            print("    ✓ 3D embodied NeuroMechFly simulation ready.")
        except Exception as exc:
            print(f"[!] 3D simulation initialization failed ({exc}); using fallback.")
            self._has_sim = False

    def step(self, escape_phase: str) -> np.ndarray:
        """Advance physics step with embodied reflex kinematics and render 3D viewport."""
        if not self._has_sim:
            # Fallback canvas
            canvas = np.zeros((self.render_h, self.render_w, 3), dtype=np.uint8)
            canvas[:] = (35, 38, 42)
            cv2.putText(
                canvas,
                "3D FLY ARENA [FALLBACK VIEWPORT]",
                (40, self.render_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (140, 160, 180),
                2,
                cv2.LINE_AA,
            )
            return canvas

        # ------------------------------------------------------------------
        # Embodied Actuator & Joint Kinematics by FSM State:
        # Strictly enforces upright root orientation [1, 0, 0, 0] and positive
        # ground clearance in +z to prevent any flipping or floor inversion.
        # ------------------------------------------------------------------
        if escape_phase == "FREEZE":
            # FREEZE: Lower center of mass towards floor, loaded spring crouch
            q = self.neutral_qpos.copy()
            q[self.root_z_idx] = self.neutral_qpos[self.root_z_idx] - 0.25  # lower COM
            for leg in ("lm", "rm", "lh", "rh"):
                fem_adr = self.leg_joints.get(f"{leg}_femur", -1)
                tib_adr = self.leg_joints.get(f"{leg}_tibia", -1)
                if fem_adr != -1:
                    q[fem_adr] += 0.15
                if tib_adr != -1:
                    q[tib_adr] += 0.20
            q[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.sim.mj_data.qpos[:] = q
            self.sim.mj_data.qvel[:] = 0.0
            self.flight_snapshot_qpos = None
            self.recovery_step = 0

        elif escape_phase == "JUMP":
            # JUMP: Ballistic extension impulse pushing down against floor, lift root Z in +z
            q = self.neutral_qpos.copy()
            # Lift fly into positive Z (+0.80 mm clearance off arena floor, never negative)
            q[self.root_z_idx] = self.neutral_qpos[self.root_z_idx] + 0.80
            # Mesothoracic (middle) & metathoracic (rear) legs extend downwards to launch
            for leg in ("lm", "rm", "lh", "rh"):
                fem_adr = self.leg_joints.get(f"{leg}_femur", -1)
                tib_adr = self.leg_joints.get(f"{leg}_tibia", -1)
                if fem_adr != -1:
                    q[fem_adr] += 0.05
                if tib_adr != -1:
                    q[tib_adr] += 0.25
            # Strictly maintain upright orientation quaternion
            q[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.sim.mj_data.qpos[:] = q
            self.sim.mj_data.qvel[:] = 0.0
            self.flight_snapshot_qpos = None
            self.recovery_step = 0

        elif escape_phase == "FLIGHT":
            # FLIGHT: Wings spread outward/upward with flutter oscillation, elevated airborne
            q = self.neutral_qpos.copy()
            q[self.root_z_idx] = self.neutral_qpos[self.root_z_idx] + 1.30  # airborne in +z
            # Retract / tuck legs aerodynamically beneath body in flight
            for leg in ("lm", "rm", "lh", "rh", "lf", "rf"):
                fem_adr = self.leg_joints.get(f"{leg}_femur", -1)
                tib_adr = self.leg_joints.get(f"{leg}_tibia", -1)
                if fem_adr != -1:
                    q[fem_adr] += 0.10
                if tib_adr != -1:
                    q[tib_adr] += 0.25
            # High-frequency wing flutter oscillation
            self.wing_flutter_phase += 0.65
            flutter = 0.15 * np.sin(self.wing_flutter_phase)
            if self.wing_joints["l_yaw"] != -1:
                q[self.wing_joints["l_yaw"]] = 0.95 + flutter
            if self.wing_joints["l_pitch"] != -1:
                q[self.wing_joints["l_pitch"]] = 0.50 + flutter * 0.5
            if self.wing_joints["r_yaw"] != -1:
                q[self.wing_joints["r_yaw"]] = -0.95 - flutter
            if self.wing_joints["r_pitch"] != -1:
                q[self.wing_joints["r_pitch"]] = 0.50 + flutter * 0.5

            q[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.sim.mj_data.qpos[:] = q
            self.sim.mj_data.qvel[:] = 0.0
            self.flight_snapshot_qpos = q.copy()
            self.recovery_step = 0

        elif escape_phase == "RECOVERY":
            # RECOVERY: Reset joints and COM smoothly back to neutral stance over 12 frames
            self.recovery_step += 1
            alpha = min(1.0, self.recovery_step / float(self.recovery_total))
            blend = 0.5 * (1.0 - np.cos(np.pi * alpha))  # smooth S-curve
            start_q = (
                self.flight_snapshot_qpos
                if self.flight_snapshot_qpos is not None
                else self.neutral_qpos
            )
            q = (1.0 - blend) * start_q + blend * self.neutral_qpos
            q[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.sim.mj_data.qpos[:] = q
            self.sim.mj_data.qvel[:] = 0.0

            if self.recovery_step >= self.recovery_total:
                self.sim.mj_data.qpos[:] = self.neutral_qpos
                self.sim.mj_data.qvel[:] = 0.0
                self.flight_snapshot_qpos = None

        else:
            # IDLE: Neutral upright standing posture
            self.sim.mj_data.qpos[:] = self.neutral_qpos
            self.sim.mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.sim.mj_data.qvel[:] = 0.0
            self.flight_snapshot_qpos = None
            self.recovery_step = 0

        # Propagate kinematics and step simulation
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.sim.step()

        # Render 3D viewport
        mj_ren = self.sim.renderer.mj_renderer
        mj_ren.update_scene(
            self.sim.mj_data, self.cam_id, self.sim.renderer.scene_option
        )
        rgb_frame = mj_ren.render()
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)


# ============================================================================
# Embodied Compositor & HUD
# ============================================================================

class EmbodiedCompositor:
    """Composites the 3D arena viewport, virtual screen inset, and ommatidia readout."""

    PHASE_COLORS = {
        "IDLE": (120, 120, 120),
        "FREEZE": (0, 200, 255),    # amber
        "JUMP": (0, 0, 255),        # bright red
        "FLIGHT": (255, 120, 0),    # cyan
        "RECOVERY": (0, 220, 80),   # green
    }

    def __init__(self, out_h: int = 540, out_w: int = 960) -> None:
        self.out_h = out_h
        self.out_w = out_w

        # Inset geometries
        # 1. Top-Right: Virtual Screen (16:9 broadcast inset)
        self.screen_w = 360
        self.screen_h = 202
        self.screen_x = out_w - self.screen_w - 20
        self.screen_y = 26

        # 2. Bottom-Right: Ommatidia Receptive Field (scaled down 25% to 160x160)
        self.omm_w = 160
        self.omm_h = 160
        self.omm_x = out_w - self.omm_w - 20
        self.omm_y = out_h - self.omm_h - 22

    def render(
        self,
        frame_3d: np.ndarray,
        raw_video_frame: np.ndarray,
        hex_image: np.ndarray,
        looming_score: float,
        escape_phase: str,
        frame_idx: int,
        timestamp_sec: float,
        total_escapes: int,
    ) -> np.ndarray:
        canvas = frame_3d.copy()
        h, w = self.out_h, self.out_w

        # ------------------------------------------------------------------
        # 1. Escape Reflex Flash Effect on Arena Viewport
        # ------------------------------------------------------------------
        if escape_phase in ("FREEZE", "JUMP"):
            flash_col = self.PHASE_COLORS[escape_phase]
            alpha = 0.12 if escape_phase == "FREEZE" else 0.22
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), flash_col, -1)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # ------------------------------------------------------------------
        # 2. Top-Right Inset: Virtual Display Screen
        # ------------------------------------------------------------------
        sx, sy = self.screen_x, self.screen_y
        sw, sh = self.screen_w, self.screen_h
        screen_sub = cv2.resize(raw_video_frame, (sw, sh))

        # Backing bezel & header
        cv2.rectangle(canvas, (sx - 4, sy - 22), (sx + sw + 4, sy + sh + 4), (18, 20, 24), -1)
        cv2.rectangle(canvas, (sx - 2, sy - 2), (sx + sw + 2, sy + sh + 2), (0, 210, 255), 2)
        canvas[sy : sy + sh, sx : sx + sw] = screen_sub

        cv2.putText(
            canvas,
            "VIRTUAL SCREEN // COEN CARR FAST BREAK",
            (sx + 4, sy - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------------------
        # 3. Bottom-Right Inset: Hexagonal Ommatidia Readout
        # ------------------------------------------------------------------
        ox, oy = self.omm_x, self.omm_y
        ow, oh = self.omm_w, self.omm_h
        omm_sub = cv2.resize(hex_image, (ow, oh), interpolation=cv2.INTER_NEAREST)

        # Backing bezel & header
        cv2.rectangle(canvas, (ox - 4, oy - 22), (ox + ow + 4, oy + oh + 4), (16, 20, 18), -1)
        cv2.rectangle(canvas, (ox - 2, oy - 2), (ox + ow + 2, oy + oh + 2), (0, 255, 120), 2)
        canvas[oy : oy + oh, ox : ox + ow] = omm_sub

        cv2.putText(
            canvas,
            "OMMATIDIA (721 FACETS)",
            (ox + 4, oy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------------------
        # 4. Bottom-Left Telemetry & Giant Fiber HUD
        # ------------------------------------------------------------------
        hud_w, hud_h = 390, 78
        hx, hy = 18, h - hud_h - 18

        # Glassmorphic backing
        hud_bg = canvas[hy : hy + hud_h, hx : hx + hud_w].copy()
        cv2.rectangle(hud_bg, (0, 0), (hud_w, hud_h), (15, 18, 22), -1)
        cv2.addWeighted(hud_bg, 0.85, canvas[hy : hy + hud_h, hx : hx + hud_w], 0.15, 0, canvas[hy : hy + hud_h, hx : hx + hud_w])
        cv2.rectangle(canvas, (hx, hy), (hx + hud_w, hy + hud_h), (80, 90, 100), 1)

        # Giant Fiber Reflex State Badge
        phase_col = self.PHASE_COLORS.get(escape_phase, (150, 150, 150))
        cv2.rectangle(canvas, (hx + 8, hy + 8), (hx + 175, hy + 34), phase_col, -1)
        cv2.putText(
            canvas,
            f"GF REFLEX: {escape_phase}",
            (hx + 14, hy + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255) if escape_phase in ("JUMP", "FLIGHT") else (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Telemetry string
        mm = int(timestamp_sec // 60)
        ss = timestamp_sec % 60
        telemetry = f"{mm:02d}:{ss:05.2f} | F:{frame_idx} | ESC:{total_escapes}"
        cv2.putText(
            canvas,
            telemetry,
            (hx + 185, hy + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 225, 240),
            1,
            cv2.LINE_AA,
        )

        # Looming Expansion Progress Bar
        bar_x = hx + 8
        bar_y = hy + 44
        bar_max_w = hud_w - 16
        bar_h_px = 22
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h_px), (35, 40, 48), -1)

        fill_w = int(np.clip(looming_score * bar_max_w * 4.5, 0, bar_max_w))
        bar_col = (
            0,
            int(np.clip(255 - looming_score * 600, 60, 255)),
            int(np.clip(looming_score * 900, 40, 255)),
        )
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_px), bar_col, -1)

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h_px), (100, 110, 125), 1)
        cv2.putText(
            canvas,
            f"LOOMING EXPANSION: {looming_score:.4f}",
            (bar_x + 8, bar_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas


# ============================================================================
# Video Encoding & Transcoding Helper
# ============================================================================

def transcode_to_h264(raw_video_path: str, h264_video_path: str) -> bool:
    """Transcodes output to standard H.264 (yuv420p) with faststart."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None

    if not ffmpeg_bin:
        print("[!] No ffmpeg binary found; keeping intermediate video output.")
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", raw_video_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "20",
        "-movflags", "+faststart",
        h264_video_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[✓] Transcoded to web-compatible H.264: {h264_video_path}")
        return True
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.decode("utf-8", errors="ignore")[-300:]
        print(f"[!] ffmpeg transcoding failed: {err_msg}")
        return False


# ============================================================================
# Main Simulation Pipeline
# ============================================================================

def run_pipeline(
    input_path: str,
    output_path: str,
    start_sec: float,
    start_frame_arg: int | None,
    num_frames: int,
) -> None:
    """Execute the embodied 3D simulation pipeline."""

    print(f"\n{'='*64}")
    print("  Drosophila Embodied 3D Compound-Eye Vision Pipeline")
    print(f"{'='*64}")
    print(f"  Input       : {input_path}")
    print(f"  Output      : {output_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    if start_frame_arg is not None:
        start_frame = start_frame_arg
    else:
        start_frame = int(round(start_sec * fps))

    if start_frame >= total_frames:
        start_frame = max(0, total_frames - num_frames)

    end_frame = min(start_frame + num_frames, total_frames)
    actual_count = end_frame - start_frame
    start_time_s = start_frame / fps

    print(
        f"  Segment     : {actual_count} frames ({start_frame} → {end_frame - 1}) "
        f"starting at {int(start_time_s//60):02d}:{start_time_s%60:05.2f} ({fps:.2f} fps)"
    )
    print(f"{'='*64}\n")

    # 1. Initialize components
    out_w, out_h = 960, 540
    print("[1/4] Initializing biological retina...")
    retina = HexRetina()

    print("[2/4] Initializing looming detector & escape reflex...")
    detector = LoomingDetector(
        retina.num_ommatidia,
        central_fraction=0.35,
        base_threshold=0.065,
        adaptation_tau=0.90,
        cooldown_frames=18,
    )
    escape_sim = EscapeReflexSimulator(
        freeze_frames=4,
        jump_frames=6,
        flight_frames=10,
        recovery_frames=12,
    )

    print("[3/4] Initializing 3D embodied NeuroMechFly simulation...")
    fly_3d = Embodied3DFlySim(render_h=out_h, render_w=out_w)
    compositor = EmbodiedCompositor(out_h=out_h, out_w=out_w)

    # 2. Output video writer
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_output_path = output_path + ".raw.mp4"

    writer = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        test_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_w, out_h))
        if test_writer.isOpened():
            writer = test_writer
            break

    if writer is None or not writer.isOpened():
        print(f"[ERROR] Cannot open output writer: {temp_output_path}")
        sys.exit(1)

    # 3. Main simulation loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    t0 = time.time()
    print("[4/4] Executing embodied simulation loop...\n")

    for i in range(actual_count):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Read failure at frame {start_frame + i}, ending.")
            break

        current_frame_idx = start_frame + i
        current_time_sec = current_frame_idx / fps

        # a) Biological retinal projection
        ommatidia = retina.raw_to_ommatidia(frame)

        # b) Looming detection
        looming_score, triggered = detector.update(ommatidia)

        # c) Escape reflex state machine
        if triggered:
            escape_sim.trigger_escape()
        phase = escape_sim.step()

        # d) Step 3D NeuroMechFly physics and render arena viewport
        arena_3d_frame = fly_3d.step(phase)

        # e) Reconstruct ommatidia visual representation
        hex_img = retina.ommatidia_to_image(ommatidia, 160, 160)

        # f) Composite 3D arena + virtual screen + ommatidia + HUD
        final_frame = compositor.render(
            arena_3d_frame,
            frame,
            hex_img,
            looming_score,
            phase,
            current_frame_idx,
            current_time_sec,
            escape_sim.total_triggers,
        )

        writer.write(final_frame)

        # Progress reporting
        if (i + 1) % 50 == 0 or i == actual_count - 1:
            elapsed = time.time() - t0
            fps_proc = (i + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (i + 1) / actual_count
            print(
                f"  [{pct:5.1f}%] frame {current_frame_idx:5d} ({int(current_time_sec//60):02d}:{current_time_sec%60:04.1f}) "
                f"looming={looming_score:.4f}  phase={phase:<10s}  ({fps_proc:.1f} fps)"
            )

    # 4. Finalization & Transcoding
    cap.release()
    writer.release()
    elapsed = time.time() - t0

    print("\n[Transcoder] Finalizing video with H.264 encoding (yuv420p)...")
    transcoded = transcode_to_h264(temp_output_path, output_path)

    if transcoded:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    else:
        if os.path.exists(temp_output_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output_path, output_path)

    print(f"\n{'='*64}")
    print(f"  ✓ Simulation complete! {actual_count} frames in {elapsed:.1f}s ({actual_count / elapsed:.1f} fps)")
    print(f"  ✓ Output video: {output_path}")
    print(f"  ✓ Total Giant Fiber escapes triggered: {escape_sim.total_triggers}")
    print(f"{'='*64}\n")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drosophila embodied 3D compound-eye simulation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", default="coen_carr.mp4",
        help="Input video path (default: coen_carr.mp4)",
    )
    parser.add_argument(
        "--output", "-o", default="output/fly_vs_carr.mp4",
        help="Output video path (default: output/fly_vs_carr.mp4)",
    )
    parser.add_argument(
        "--start-sec", type=float, default=140.0,
        help="Start time in seconds (default: 140.0 = 02:20)",
    )
    parser.add_argument(
        "--start", "-s", type=int, default=None,
        help="Start frame index (overrides --start-sec)",
    )
    parser.add_argument(
        "--count", "-n", type=int, default=600,
        help="Number of frames to process (default: 600)",
    )
    parser.add_argument(
        "--duration-sec", type=float, default=None,
        help="Duration in seconds (overrides --count)",
    )

    args = parser.parse_args()

    num_frames = args.count
    if args.duration_sec is not None:
        num_frames = int(args.duration_sec * 24.0)

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        start_sec=args.start_sec,
        start_frame_arg=args.start,
        num_frames=num_frames,
    )


if __name__ == "__main__":
    main()
