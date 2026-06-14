---
name: no-face-edge-cases
description: Distinguishing deliberate face covering from tracking loss, and what to show
metadata:
  type: project
---

When MediaPipe loses the face, there are two very different scenarios that need different handling:

1. **Deliberate face covering** (hand over face, mask, etc.) — should show a "covered face" frame/state rather than freezing or going blank. This is a deliberate expression.

2. **Tracking loss at extreme angles** — face rotated beyond ~30° where MediaPipe drops detection. In this case, the head direction from the last known pose should persist briefly, and eye/mouth should already be in SIM mode.

3. **Actual no-face** (user walked away) — should eventually fade/hold last state gracefully.

Currently all three cases result in `face_found=False` → last known state frozen. Need a timeout-based approach:
- Short loss (<0.5s): hold last state (momentary occlusion)
- Medium loss (0.5-2s): transition to covered/hidden state
- Long loss (>2s): fade/hold with indication

**Why:** The current behavior of simply freezing the last state looks wrong when the user deliberately covers their face or turns too far.

**How to apply:** Add a face-loss timer in tracker or main loop. Distinguish cases by loss duration. [[dual-mode-tuning]]
