---
name: Standalone Streamlit preview routing
description: A port-open workflow can still be unreachable through the project root preview in an artifact-routed workspace.
---

In this artifact-routed workspace, a standalone Streamlit workflow and a validated `.replit` port mapping did not create a root preview route; a registered root web artifact was required. Managed artifact services start from their artifact directory, so a root-level app needs an explicit `cd ../..` before launching. Removing a workflow definition also may leave its old server process holding the port.

**Why:** Workflow port detection, application routing, and process lifecycle are separate. A healthy server is not enough if the root artifact route is missing, and a stale process can prevent the managed replacement from binding.

**How to apply:** For this project, keep the web artifact at `/`, bind Streamlit to its managed `PORT` on `0.0.0.0`, launch from the workspace root, and check for leftover listeners before restarting. Verify both app health and the user-facing Preview.
