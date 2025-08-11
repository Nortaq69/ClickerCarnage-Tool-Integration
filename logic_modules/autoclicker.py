import threading
import time
import random
from pynput.mouse import Controller, Button
from pynput import keyboard

class AutoClicker:
    """
    Advanced AutoClicker supporting:
    - Adjustable CPS
    - Interval randomization
    - Click position randomization (within area)
    - Click patterns: burst, hold, toggle
    - Multi-point clicking
    - Hotkey start/stop
    """
    def __init__(self, cps=10, button='left', hotkey='f6',
                 randomize_interval=False, min_cps=None, max_cps=None,
                 randomize_position=False, area=None,
                 pattern='toggle', burst_count=5, burst_pause=0.5,
                 multipoints=None):
        self.cps = cps
        self.button = button
        self.hotkey = hotkey
        self.randomize_interval = randomize_interval
        self.min_cps = min_cps or cps
        self.max_cps = max_cps or cps
        self.randomize_position = randomize_position
        self.area = area  # (x1, y1, x2, y2)
        self.pattern = pattern  # 'toggle', 'burst', 'hold'
        self.burst_count = burst_count
        self.burst_pause = burst_pause
        self.multipoints = multipoints or []  # [(x, y), ...]
        self.running = False
        self.thread = None
        self.mouse = Controller()
        self.listener = None

    def _get_next_delay(self):
        if self.randomize_interval:
            cps = random.uniform(self.min_cps, self.max_cps)
            return 1.0 / cps
        return 1.0 / self.cps

    def _get_next_position(self):
        if self.randomize_position and self.area:
            x1, y1, x2, y2 = self.area
            x = random.randint(x1, x2)
            y = random.randint(y1, y2)
            return (x, y)
        return None

    def _click(self):
        btn = Button.left if self.button == 'left' else Button.right if self.button == 'right' else Button.middle
        if self.pattern == 'burst':
            while self.running:
                for _ in range(self.burst_count):
                    if not self.running:
                        break
                    self._do_click(btn)
                    time.sleep(self._get_next_delay())
                time.sleep(self.burst_pause)
        elif self.pattern == 'hold':
            while self.running:
                self._do_click(btn)
                time.sleep(self._get_next_delay())
        elif self.pattern == 'toggle':
            while self.running:
                self._do_click(btn)
                time.sleep(self._get_next_delay())

    def _do_click(self, btn):
        if self.multipoints:
            for x, y in self.multipoints:
                self.mouse.position = (x, y)
                self.mouse.click(btn)
        else:
            pos = self._get_next_position()
            if pos:
                self.mouse.position = pos
            self.mouse.click(btn)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._click, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
            self.thread = None

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def listen_hotkey(self):
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char and key.char.lower() == self.hotkey.lower():
                    self.toggle()
            except Exception:
                pass
        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

# Example usage for manual test
if __name__ == '__main__':
    ac = AutoClicker(
        cps=15,
        button='left',
        hotkey='f6',
        randomize_interval=True,
        min_cps=10,
        max_cps=20,
        randomize_position=True,
        area=(500, 500, 600, 600),
        pattern='burst',
        burst_count=10,
        burst_pause=0.3,
        multipoints=[(600, 600), (700, 700)]
    )
    print('AutoClicker ready. Press F6 to start/stop.')
    ac.listen_hotkey()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        ac.stop()
        print('Stopped.') 