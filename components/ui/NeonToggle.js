// NeonToggle.js
// Creates a neon-styled, animated toggle switch with SVG glow

export default class NeonToggle {
    constructor({ checked = false, onChange = null, accent = '#00fff7', width = 60, height = 34 }) {
        this.checked = checked;
        this.onChange = onChange;
        this.accent = accent;
        this.width = width;
        this.height = height;
        this.toggle = this.createToggle();
    }

    createToggle() {
        const wrapper = document.createElement('div');
        wrapper.className = 'neon-toggle-wrapper';
        wrapper.style.width = this.width + 'px';
        wrapper.style.height = this.height + 'px';
        wrapper.innerHTML = `
            <svg width="60" height="34" viewBox="0 0 60 34" fill="none" xmlns="http://www.w3.org/2000/svg">
              <g filter="url(#glow)">
                <rect x="2" y="8" width="56" height="18" rx="9" fill="#111" stroke="${this.accent}" stroke-width="4"/>
                <circle cx="${this.checked ? 43 : 17}" cy="17" r="9" fill="${this.accent}"/>
              </g>
              <defs>
                <filter id="glow" x="0" y="0" width="60" height="34" filterUnits="userSpaceOnUse">
                  <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>
            </svg>
        `;
        wrapper.addEventListener('click', () => {
            this.checked = !this.checked;
            this.update();
            if (this.onChange) this.onChange(this.checked);
        });
        return wrapper;
    }

    update() {
        const circle = this.toggle.querySelector('circle');
        circle.setAttribute('cx', this.checked ? 43 : 17);
    }

    mount(parent) {
        parent.appendChild(this.toggle);
    }

    getElement() {
        return this.toggle;
    }
} 