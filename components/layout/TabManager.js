// TabManager.js
// Handles animated, neon-styled tab navigation

export default class TabManager {
    constructor(tabs, container) {
        this.tabs = tabs; // [{label, icon, content}]
        this.container = container;
        this.active = 0;
        this.tabBar = this.createTabBar();
        this.contentArea = document.createElement('div');
        this.contentArea.className = 'tab-content-area';
        this.container.appendChild(this.tabBar);
        this.container.appendChild(this.contentArea);
        this.renderContent();
    }

    createTabBar() {
        const bar = document.createElement('div');
        bar.className = 'tab-bar';
        this.tabs.forEach((tab, i) => {
            const btn = document.createElement('button');
            btn.className = 'tab-btn' + (i === this.active ? ' active' : '');
            btn.innerHTML = `${tab.icon ? `<img src="${tab.icon}" class="tab-icon"/>` : ''}<span>${tab.label}</span>`;
            btn.addEventListener('click', () => this.setActive(i));
            bar.appendChild(btn);
        });
        return bar;
    }

    setActive(i) {
        this.active = i;
        this.tabBar.querySelectorAll('.tab-btn').forEach((btn, idx) => {
            btn.classList.toggle('active', idx === i);
        });
        this.renderContent();
    }

    renderContent() {
        this.contentArea.innerHTML = '';
        this.contentArea.appendChild(this.tabs[this.active].content);
    }
} 