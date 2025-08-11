// IconLoader.js
// Loads SVG icons from assets/ui and injects as inline SVG for neon coloring/animation

export default class IconLoader {
    static async load(iconName) {
        const res = await fetch(`assets/ui/${iconName}.svg`);
        if (!res.ok) throw new Error('Icon not found: ' + iconName);
        const svgText = await res.text();
        const div = document.createElement('div');
        div.innerHTML = svgText;
        return div.firstElementChild;
    }
} 