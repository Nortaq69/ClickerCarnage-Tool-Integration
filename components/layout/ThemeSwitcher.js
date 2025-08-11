// ThemeSwitcher.js
// Handles neon accent color and dark/light glassmorphism background switching

export default class ThemeSwitcher {
    constructor({ accents = ['#00fff7', '#ff00ff', '#2ed573', '#ff4757'], defaultAccent = 0, onChange = null }) {
        this.accents = accents;
        this.current = defaultAccent;
        this.onChange = onChange;
        this.switcher = this.createSwitcher();
    }

    createSwitcher() {
        const wrapper = document.createElement('div');
        wrapper.className = 'theme-switcher';
        this.accents.forEach((color, i) => {
            const btn = document.createElement('button');
            btn.className = 'theme-accent-btn' + (i === this.current ? ' active' : '');
            btn.style.background = color;
            btn.onclick = () => this.setAccent(i);
            wrapper.appendChild(btn);
        });
        return wrapper;
    }

    setAccent(i) {
        this.current = i;
        document.documentElement.style.setProperty('--accent', this.accents[i]);
        this.switcher.querySelectorAll('.theme-accent-btn').forEach((btn, idx) => {
            btn.classList.toggle('active', idx === i);
        });
        if (this.onChange) this.onChange(this.accents[i]);
    }

    mount(parent) {
        parent.appendChild(this.switcher);
    }

    getElement() {
        return this.switcher;
    }
} 