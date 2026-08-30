"""VTB Demo V1 — dual-window display, special states (leaving/returning,
hand-on-face, drinking, lean-forward)."""

from __future__ import annotations

import time

import pygame

from audio_capture import AudioCapture
import config
from display import FrameDisplay
from mapper import StateMapper
from simulator import EyeSimulator, MouthSimulator
from tracker import FaceTracker


class Debounced:
    """Frame-count Schmitt trigger: N consecutive frames to enter, M to exit."""

    def __init__(self, enter_frames: int, exit_frames: int) -> None:
        self._enter = enter_frames
        self._exit = exit_frames
        self._count = 0
        self.active = False

    def update(self, raw: bool) -> bool:
        if raw == self.active:
            self._count = 0
            return self.active
        self._count += 1
        needed = self._enter if not self.active else self._exit
        if self._count >= needed:
            self.active = raw
            self._count = 0
        return self.active


def main() -> None:
    tracker = FaceTracker()
    mapper = StateMapper()
    display = FrameDisplay()

    audio = AudioCapture()
    eye_sim = EyeSimulator()
    mouth_sim = MouthSimulator()

    # Leaving / returning — both one-shot actions with confirmation intervals
    face_lost_at: float | None = None
    face_found_since: float | None = None
    returning_until: float = 0.0
    leaving_state: str = "default"  # default | leaving | returning

    # Debounced special states (auto-detected)
    hof = Debounced(
        config.HAND_ON_FACE_ENTER_FRAMES, config.HAND_ON_FACE_EXIT_FRAMES
    )
    drink = Debounced(config.DRINKING_ENTER_FRAMES, config.DRINKING_EXIT_FRAMES)
    lean = Debounced(config.LEAN_ENTER_FRAMES, config.LEAN_EXIT_FRAMES)
    manual_hof = False  # K_h toggle override

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
                    # Manual hand-on-face toggle
                    if event.key == pygame.K_h:
                        manual_hof = not manual_hof
                        print(f"[Theme] manual hand_on_face = {manual_hof}")

            # ---- tracking ----
            if not calibration_notified and tracker.calibrated:
                print("[Ready] Calibration complete.")
                calibration_notified = True

            tracking_state = tracker.read_state()

            # ---- leaving / returning state machine (one-shot actions) ----
            # Each action fires exactly once, only after its confirmation
            # interval; single-frame noise neither fires nor re-fires them.
            if not tracking_state.face_found:
                face_found_since = None
                if face_lost_at is None:
                    face_lost_at = time.time()
                elif (
                    time.time() - face_lost_at > config.LEAVING_TIMEOUT
                    and leaving_state == "default"
                ):
                    leaving_state = "leaving"
                    print("[State] leaving (action)")
            else:
                if face_found_since is None:
                    face_found_since = time.time()
                confirmed_back = (
                    time.time() - face_found_since
                    >= config.RETURN_CONFIRM_DURATION
                )
                if leaving_state == "leaving" and confirmed_back:
                    leaving_state = "returning"
                    returning_until = time.time() + config.RETURNING_DURATION
                    print("[State] returning (action)")
                elif (
                    leaving_state == "returning"
                    and time.time() >= returning_until
                ):
                    leaving_state = "default"
                    print("[State] default")
                face_lost_at = None

            # ---- special-state raw signals + debounce ----
            face_ok = tracking_state.face_found
            hof_active = hof.update(face_ok and tracker.hand_on_face)
            drink_active = drink.update(face_ok and tracker.hand_at_mouth)
            # Lean is a STATE: Schmitt band on the scale signal (enter high,
            # exit low) on top of the frame-count debounce.
            lean_threshold = (
                config.LEAN_EXIT_SCALE if lean.active
                else config.LEAN_SCALE_THRESHOLD
            )
            lean_active = lean.update(
                face_ok
                and tracker.calibrated
                and tracking_state.face_scale >= lean_threshold
            )

            # ---- theme resolution (recomputed every frame, no stuck states) ----
            # Priority: leaving > returning > drinking > hand_on_face > lean
            if leaving_state == "leaving":
                theme = "leaving"
            elif leaving_state == "returning":
                theme = "returning"
            elif drink_active:
                theme = config.THEME_DRINKING
            elif hof_active or manual_hof:
                theme = config.THEME_HAND_ON_FACE
            elif lean_active:
                theme = config.THEME_LEAN_FORWARD
            else:
                theme = config.THEME_DEFAULT
            mapper.theme = theme

            # ---- dual-mode: CAM vs SIM ----
            confidence = tracking_state.face_confidence
            if (
                leaving_state in ("leaving", "returning")
                or confidence < config.FACE_CONFIDENCE_THRESHOLD
            ):
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
            target_discrete_state = mapper.classify(tracking_state)
            target_state_key, target_frame_path = mapper.resolve_frame(
                target_discrete_state
            )

            # ---- momentary face loss: hold last frame ----
            if (
                not tracking_state.face_found
                and leaving_state == "default"
            ):
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
                "hand_landmarks": tracker.last_hand_landmarks,
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
                "mouth_raw": float(tracking_state.mouth_raw),
                "face_scale": float(tracking_state.face_scale),
                "hand_on_face": hof_active,
                "drinking": drink_active,
                "lean_forward": lean_active,
                "face_box": tracker.face_box,
                "mouth_zone": tracker.mouth_zone,
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
