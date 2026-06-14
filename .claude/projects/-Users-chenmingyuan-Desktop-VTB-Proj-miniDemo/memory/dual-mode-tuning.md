---
name: dual-mode-tuning
description: SIM/CAM switching threshold feels too sensitive, needs further calibration
metadata:
  type: feedback
---

Dual-mode switching (camera vs simulator for eyes/mouth) currently triggers a bit too easily at moderate head angles. The _actual_ SIM behavior once triggered is fine—the issue is the boundary between CAM and SIM needs to be pushed further (higher confidence required for CAM, or a smoother blend).

Current threshold: `FACE_CONFIDENCE_THRESHOLD = 0.35` with weights `0.3×rot + 0.3×ar + 0.4×eye_width_sym`.

**Why:** The eye-width symmetry signal (ew) drops faster than expected at moderate angles, triggering SIM earlier than ideal. Need to either raise threshold, adjust ew clamping range, or add hysteresis to the CAM→SIM transition.

**How to apply:** Adjust `FACE_CONFIDENCE_THRESHOLD` or the `eye_width_sym` clamping parameters in `config.py`. Consider adding a separate hysteresis for the CAM↔SIM transition (different from the Schmitt triggers used for eye/mouth state classification). [[no-face-edge-cases]]
