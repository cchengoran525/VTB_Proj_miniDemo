---
name: future-features
description: Features deferred past MVP v1 — hand tracking, object interaction, audio-driven mouth
metadata:
  type: project
---

Features explicitly deferred past the first MVP:

- **Hand tracking** — detect when hand covers face, hand gestures, etc. V1 just needs to handle "face lost" gracefully.
- **Object interaction tracking** — props, items held near face that may occlude landmarks.
- **Audio-driven mouth** — the `MouthSimulator` placeholder will be replaced with actual microphone amplitude → mouth openness mapping. Infrastructure is ready (simulator has `update()` interface that can be swapped).

**Why:** These add significant complexity. Face + eye + mouth + head tracking need to be solid first.

**How to apply:** After the core pipeline (dual-mode, 5×5 grid, anti-jitter) is stable, tackle in this order: (1) audio mouth, (2) hand detection for face-covering, (3) object interaction. [[no-face-edge-cases]] [[dual-mode-tuning]]
