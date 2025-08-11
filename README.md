# ClickerCarnage Tool Integration

**ClickerCarnage** is a galaxy-themed modular automation and clicker suite. It features a cosmic UI, plugin system, and automation tools for power users and tinkerers.

## 🚀 Features
- Modular plugin system (auto-detects plugins in `logic_modules/`)
- Galaxy/cosmic themed UI
- Native Python GUI (PySide6)
- Launchable from the NexusHub suite
- Status/log area and plugin list
- Future: Web bridge integration for remote control

## 🛠️ How to Use
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch from NexusHub:**
   - Open the NexusHub suite and go to the ClickerCarnage tool.
   - Click the **Launch ClickerCarnage** button.
   - The native GUI will open.
   - Detected plugins will be listed in the web UI.

3. **Add Plugins:**
   - Drop `.py` plugin files into `logic_modules/`.
   - Plugins are auto-detected and loaded on launch.

## 🔮 Roadmap
- Web bridge for remote control and dashboard
- Plugin management from the web UI
- Real-time status sync between web and native app

## 🪐 About
ClickerCarnage is designed for automation, rapid prototyping, and cosmic fun. All code is MIT licensed. For more, see the main ClickerCarnage README. 