from __future__ import annotations

from pathlib import Path

from display import FrameDisplay
from mapper import StateMapper
from tracker import FaceTracker


def main() -> None:
    tracker = FaceTracker()
    mapper = StateMapper()
    display = FrameDisplay()

    try:
        default_state_key, default_frame_path = mapper.resolve_frame(
            mapper.classify(tracker.last_state)
        )
        display.render(default_frame_path, default_state_key)

        while True:
            if display.should_quit():
                break

            tracking_state = tracker.read_state()
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

