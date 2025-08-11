// Modal.js
// Animated, neon-styled modal dialog

export default class Modal {
    constructor({ content = '', onClose = null, accent = '#00fff7' }) {
        this.content = content;
        this.onClose = onClose;
        this.accent = accent;
        this.modal = this.createModal();
    }

    createModal() {
        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        overlay.innerHTML = `
            <div class="modal-dialog" style="border-color:${this.accent}">
                <button class="modal-close">&times;</button>
                <div class="modal-content">${typeof this.content === 'string' ? this.content : ''}</div>
            </div>
        `;
        overlay.querySelector('.modal-close').onclick = () => this.close();
        overlay.addEventListener('click', e => {
            if (e.target === overlay) this.close();
        });
        if (typeof this.content !== 'string') {
            overlay.querySelector('.modal-content').appendChild(this.content);
        }
        return overlay;
    }

    open() {
        document.body.appendChild(this.modal);
        setTimeout(() => this.modal.classList.add('open'), 10);
    }

    close() {
        this.modal.classList.remove('open');
        setTimeout(() => {
            if (this.modal.parentNode) this.modal.parentNode.removeChild(this.modal);
            if (this.onClose) this.onClose();
        }, 300);
    }
} 