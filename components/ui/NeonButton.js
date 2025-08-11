// NeonButton.js
// Creates a neon-styled, animated button with SVG glow and ripple effects

export default class NeonButton {
    constructor({ label = '', icon = null, onClick = null, accent = '#00fff7', width = 180, height = 56 }) {
        this.label = label;
        this.icon = icon;
        this.onClick = onClick;
        this.accent = accent;
        this.width = width;
        this.height = height;
        this.button = this.createButton();
    }

    createButton() {
        const btn = document.createElement('button');
        btn.className = 'neon-btn';
        btn.style.width = this.width + 'px';
        btn.style.height = this.height + 'px';
        btn.innerHTML = `
            <span class="neon-btn-bg"></span>
            <span class="neon-btn-content">
                ${this.icon ? `<img src="${this.icon}" class="neon-btn-icon"/>` : ''}
                <span class="neon-btn-label">${this.label}</span>
            </span>
            <span class="neon-btn-glow"></span>
        `;
        btn.addEventListener('click', e => {
            btn.classList.add('active');
            setTimeout(() => btn.classList.remove('active'), 200);
            if (this.onClick) this.onClick(e);
        });
        btn.addEventListener('mouseenter', () => btn.classList.add('hover'));
        btn.addEventListener('mouseleave', () => btn.classList.remove('hover'));
        return btn;
    }

    mount(parent) {
        parent.appendChild(this.button);
    }

    getElement() {
        return this.button;
    }
} 