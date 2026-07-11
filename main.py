"""VTB Demo V1 — dual‑window display, theme toggle, leaving/returning."""

from __future__ import annotations

import random
import time
from pathlib import Path

import pygame

from audio_capture import AudioCapture
import config
from display import FrameDisplay
from mapper import StateMapper
from simulator import EyeSimulator, MouthSimulator
from tracker import FaceTracker


def main() -> None:
    tracker = FaceTracker()
    mapper = StateMapper()
    display = FrameDisplay()

    audio = AudioCapture()
    eye_sim = EyeSimulator()
    mouth_sim = MouthSimulator()

    # Leaving / returning state machine
    face_lost_at: float | None = None
    returning_until: float = 0.0
    leaving_state: str = "default"  # default | leaving | returning

    # Auto hand-on-face detection (mouth_raw spike)
    hof_counter: int = 0
    hof_cooldown: int = 0

    try:
        default_state_key, default_frame_path = mapper.resolve_frame(
            mapper.classify(tracker.last_state)
        )
        display.render(default_frame_path, default_state_key)

        calibration_notified = False
        clock = pygame.time.Clock()

        while True:
            dt = clock.tick(config.TARGET_FPS) / 1000.0

            # ---- handle input ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        raise SystemExit
                    # Theme toggle
                    if event.key == pygame.K_h:
                        if mapper.theme == config.THEME_DEFAULT:
                            mapper.theme = config.THEME_HAND_ON_FACE
                            print("[Theme] hand_on_face")
                        else:
                            mapper.theme = config.THEME_DEFAULT
                            print("[Theme] default")

            # ---- tracking ----
            if not calibration_notified and tracker.calibrated:
                print("[Ready] Calibration complete.")
                calibration_notified = True

            tracking_state = tracker.read_state()

            # ---- leaving / returning state machine ----
            if not tracking_state.face_found:
                if face_lost_at is None:
                    face_lost_at = time.time()
                elif time.time() - face_lost_at > config.LEAVING_TIMEOUT:
                    if leaving_state == "default":
                        leaving_state = "leaving"
                        print("[State] leaving")
            else:
                if leaving_state == "leaving":
                    leaving_state = "returning"
                    returning_until = time.time() + config.RETURNING_DURATION
                    print("[State] returning")
                elif (
                    leaving_state == "returning"
                    and time.time() >= returning_until
                ):
                    leaving_state = "default"
                    print("[State] default")
                face_lost_at = None

            # ---- dual-mode: CAM vs SIM ----
            confidence = tracking_state.face_confidence

            # ---- auto hand_on_face (mouth_raw spike → hand near face) ----
            if hof_cooldown > 0:
                hof_cooldown -= 1
            if tracking_state.mouth_raw > 0.30:
                hof_counter += 1
                if hof_counter >= 5 and hof_cooldown <= 0:
                    if mapper.theme == config.THEME_DEFAULT:
                        mapper.theme = config.THEME_HAND_ON_FACE
                        print("[Theme] hand_on_face (auto)")
                        hof_cooldown = 60  # 2s cooldown
            else:
                hof_counter = max(0, hof_counter - 1)
                if hof_counter <= 0 and mapper.theme == config.THEME_HAND_ON_FACE:
                    mapper.theme = config.THEME_DEFAULT
                    print("[Theme] default (auto)")

            if leaving_state in ("leaving", "returning"):
                # Override: use SIM during leaving/returning
                eye_s = eye_sim.update(dt)
                mouth_s = mouth_sim.update(dt, audio.amplitude)
                tracking_state.left_eye_open = {
                    "open": 0.95, "half": 0.55, "closed": 0.15,
                }[eye_s]
                tracking_state.right_eye_open = tracking_state.left_eye_open
                tracking_state.mouth_open = {
                    "closed": 0.05, "half": 0.30, "open": 0.80,
                }[mouth_s]
            elif confidence < config.FACE_CONFIDENCE_THRESHOLD:
                eye_s = eye_sim.update(dt)
                mouth_s = mouth_sim.update(dt, audio.amplitude)
                tracking_state.left_eye_open = {
                    "open": 0.95, "half": 0.55, "closed": 0.15,
                }[eye_s]
                tracking_state.right_eye_open = tracking_state.left_eye_open
                tracking_state.mouth_open = {
                    "closed": 0.05, "half": 0.30, "open": 0.80,
                }[mouth_s]
            else:
                eye_sim.reset()
                mouth_sim.reset()

            # ---- mapping ----
            mapper.theme = (
                config.THEME_DEFAULT
                if leaving_state == "default"
                else leaving_state
            )
            target_discrete_state = mapper.classify(tracking_state)
            target_state_key, target_frame_path = mapper.resolve_frame(
                target_discrete_state
            )

            # ---- foreground / leave detection ----
            if (
                not tracking_state.face_found
                and leaving_state == "default"
            ):
                # Momentary face loss — hold last frame
                if display.current_frame_path is None:
                    display.current_frame_path = default_frame_path
                target_frame_path = display.current_frame_path
                target_state_key = (
                    display.current_state_key or default_state_key
                )

            transition_frame_path = (
                mapper.get_transition_frame(
                    display.current_state_key, target_state_key
                )
                if config.TRANSITION_FRAMES_ENABLED
                else None
            )

            next_frame_path, next_state_key, _reason = display.choose_next_frame(
                target_state_key=target_state_key,
                target_frame_path=target_frame_path,
                transition_frame_path=transition_frame_path,
                available_state_keys=mapper.available_state_keys(),
                get_frame_path=lambda key: mapper.state_frames[key],
            )

            # ---- debug info ----
            debug = {
                "camera_frame": tracker.last_frame,
                "landmarks": tracker.last_landmarks,
                "state_key": next_state_key,
                "head": target_discrete_state.head,
                "mouth": target_discrete_state.mouth,
                "eye": target_discrete_state.eye,
                "theme": target_discrete_state.theme,
                "mode": (
                    "leaving"
                    if leaving_state == "leaving"
                    else "returning"
                    if leaving_state == "returning"
                    else "SIM"
                    if confidence < config.FACE_CONFIDENCE_THRESHOLD
                    else "CAM"
                ),
                "conf": confidence,
                "yaw_deg": float(tracking_state.yaw * 57.3),
                "pitch_deg": float(tracking_state.pitch * 57.3),
            }

            display.render(
                next_frame_path, next_state_key,
                theme=target_discrete_state.theme,
                debug=debug,
            )
    except SystemExit:
        pass
    finally:
        tracker.close()
        display.close()
        audio.close()


if __name__ == "__main__":
    main()
