from __future__ import annotations

import math
from pathlib import Path

import config
from display import FrameDisplay
from mapper import StateMapper
from simulator import EyeSimulator, MouthSimulator
from tracker import FaceTracker


def main() -> None:
    tracker = FaceTracker()
    mapper = StateMapper()
    display = FrameDisplay()

    eye_sim = EyeSimulator()
    mouth_sim = MouthSimulator()

    try:
        default_state_key, default_frame_path = mapper.resolve_frame(
            mapper.classify(tracker.last_state)
        )
        display.render(default_frame_path, default_state_key)

        calibration_notified = False

        while True:
            if display.should_quit():
                break

            if not calibration_notified and tracker.calibrated:
                print("[Ready] Calibration complete — head tracking active.")
                calibration_notified = True

            tracking_state = tracker.read_state()

            # ---- dual-mode: switch eye / mouth to simulator when confidence low ----
            confidence = tracking_state.face_confidence

            if confidence < config.FACE_CONFIDENCE_THRESHOLD:
                # Low confidence (side profile, etc.) → simulated eye / mouth
                eye_s = eye_sim.update(1.0 / config.TARGET_FPS)
                mouth_s = mouth_sim.update(1.0 / config.TARGET_FPS)

                tracking_state.left_eye_open = {
                    "open": 0.95, "half": 0.55, "closed": 0.15,
                }[eye_s]
                tracking_state.right_eye_open = tracking_state.left_eye_open
                tracking_state.mouth_open = 0.8 if mouth_s == "open" else 0.05
            else:
                # High confidence → keep camera values, reset simulators
                eye_sim.reset()
                mouth_sim.reset()

            # ---- classification & display ----
            target_discrete_state = mapper.classify(tracking_state)
            target_state_key, target_frame_path = mapper.resolve_frame(target_discrete_state)
            transition_frame_path = mapper.get_transition_frame(
                display.current_state_key, target_state_key
            )

            next_frame_path, next_state_key, _reason = display.choose_next_frame(
                target_state_key=target_state_key,
                target_frame_path=target_frame_path,
                transition_frame_path=transition_frame_path,
                available_state_keys=mapper.available_state_keys(),
                get_frame_path=lambda key: mapper.state_frames[key],
            )
            display.render(next_frame_path, next_state_key)
            display.tick()
    finally:
        tracker.close()
        display.close()


if __name__ == "__main__":
    main()
