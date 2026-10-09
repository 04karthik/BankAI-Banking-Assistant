---
name: Standalone Streamlit preview routing
description: A port-open workflow can still be unreachable through the project root preview in an artifact-routed workspace.
---

In this artifact-routed workspace, the standalone Streamlit workflow bound to `0.0.0.0:5000` and returned a healthy response directly, while the project root proxy continued returning 404. Adding a validated `.replit` port mapping did not make the root proxy route resolve.

**Why:** Workflow port detection and the application router's browser-visible routing are separate checks; the workspace root had no registered Streamlit artifact.

**How to apply:** Verify both the app's own HTTP response and the user-facing project preview. Do not report the root preview as working based only on a running workflow or open port.
