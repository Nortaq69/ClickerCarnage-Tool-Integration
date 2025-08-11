// NeonGauge.js
// Creates a neon-styled, animated circular gauge for real-time stats

export default class NeonGauge {
    constructor({ value = 0, min = 0, max = 100, label = '', accent = '#00fff7', size = 120 }) {
        this.value = value;
        this.min = min;
        this.max = max;
        this.label = label;
        this.accent = accent;
        this.size = size;
        this.gauge = this.createGauge();
    }

    createGauge() {
        const wrapper = document.createElement('div');
        wrapper.className = 'neon-gauge-wrapper';
        wrapper.style.width = wrapper.style.height = this.size + 'px';
        const radius = (this.size / 2) - 12;
        const circumference = 2 * Math.PI * radius;
        wrapper.innerHTML = `
            <svg width="${this.size}" height="${this.size}" viewBox="0 0 ${this.size} ${this.size}" fill="none" xmlns="http://www.w3.org/2000/svg">
                <g filter="url(#glow)">
                    <circle cx="${this.size/2}" cy="${this.size/2}" r="${radius}" stroke="#222" stroke-width="12"/>
                    <circle class="gauge-bar" cx="${this.size/2}" cy="${this.size/2}" r="${radius}" stroke="${this.accent}" stroke-width="12" stroke-linecap="round" fill="none" stroke-dasharray="${circumference}" stroke-dashoffset="${circumference}"/>
                </g>
                <defs>
                    <filter id="glow" x="0" y="0" width="${this.size}" height="${this.size}" filterUnits="userSpaceOnUse">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                        <feMerge>
                            <feMergeNode in="coloredBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>
            </svg>
            <div class="gauge-label">${this.label}</div>
            <div class="gauge-value">${this.value}</div>
        `;
        this.bar = wrapper.querySelector('.gauge-bar');
        this.valueEl = wrapper.querySelector('.gauge-value');
        this.circumference = circumference;
        this.radius = radius;
        this.setValue(this.value);
        return wrapper;
    }

    setValue(val) {
        this.value = Math.max(this.min, Math.min(this.max, val));
        const percent = (this.value - this.min) / (this.max - this.min);
        const offset = this.circumference * (1 - percent);
        this.bar.style.strokeDashoffset = offset;
        this.valueEl.textContent = Math.round(this.value);
    }

    animateTo(val, duration = 600) {
        const start = this.value;
        const end = Math.max(this.min, Math.min(this.max, val));
        const startTime = performance.now();
        const animate = (now) => {
            const elapsed = now - startTime;
            const t = Math.min(1, elapsed / duration);
            const current = start + (end - start) * t;
            this.setValue(current);
            if (t < 1) requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }

    mount(parent) {
        parent.appendChild(this.gauge);
    }

    getElement() {
        return this.gauge;
    }
} 