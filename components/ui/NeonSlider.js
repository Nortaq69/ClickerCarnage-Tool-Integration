// NeonSlider.js
// Creates a neon-styled, animated slider with SVG glow

export default class NeonSlider {
    constructor({ min = 0, max = 100, value = 50, onChange = null, accent = '#00fff7', width = 220 }) {
        this.min = min;
        this.max = max;
        this.value = value;
        this.onChange = onChange;
        this.accent = accent;
        this.width = width;
        this.slider = this.createSlider();
    }

    createSlider() {
        const wrapper = document.createElement('div');
        wrapper.className = 'neon-slider-wrapper';
        wrapper.style.width = this.width + 'px';
        wrapper.innerHTML = `
            <svg width="${this.width}" height="36" viewBox="0 0 ${this.width} 36" fill="none" xmlns="http://www.w3.org/2000/svg">
              <g filter="url(#glow)">
                <rect x="10" y="16" width="${this.width-20}" height="4" rx="2" fill="#111" stroke="${this.accent}" stroke-width="2"/>
                <circle cx="${this.valueToX(this.value)}" cy="18" r="12" fill="${this.accent}"/>
              </g>
              <defs>
                <filter id="glow" x="0" y="0" width="${this.width}" height="36" filterUnits="userSpaceOnUse">
                  <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>
            </svg>
        `;
        const svg = wrapper.querySelector('svg');
        const thumb = svg.querySelector('circle');
        let dragging = false;
        thumb.addEventListener('mousedown', e => {
            dragging = true;
            document.body.style.userSelect = 'none';
        });
        document.addEventListener('mousemove', e => {
            if (!dragging) return;
            const rect = svg.getBoundingClientRect();
            let x = e.clientX - rect.left;
            x = Math.max(22, Math.min(this.width-22, x));
            this.value = this.xToValue(x);
            thumb.setAttribute('cx', this.valueToX(this.value));
            if (this.onChange) this.onChange(this.value);
        });
        document.addEventListener('mouseup', () => {
            dragging = false;
            document.body.style.userSelect = '';
        });
        return wrapper;
    }

    valueToX(value) {
        return 22 + ((value - this.min) / (this.max - this.min)) * (this.width - 44);
    }

    xToValue(x) {
        return this.min + ((x - 22) / (this.width - 44)) * (this.max - this.min);
    }

    mount(parent) {
        parent.appendChild(this.slider);
    }

    getElement() {
        return this.slider;
    }
} 