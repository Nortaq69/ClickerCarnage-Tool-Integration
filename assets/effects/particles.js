// particles.js
// Neon cyberpunk particle background effect

export default function startParticles({ color = '#00fff7', count = 60 } = {}) {
    const canvas = document.createElement('canvas');
    canvas.className = 'bg-particles-canvas';
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    let w = window.innerWidth, h = window.innerHeight;
    canvas.width = w; canvas.height = h;
    window.addEventListener('resize', () => {
        w = window.innerWidth; h = window.innerHeight;
        canvas.width = w; canvas.height = h;
    });
    const particles = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: 1 + Math.random() * 2,
        vx: (Math.random() - 0.5) * 0.7,
        vy: (Math.random() - 0.5) * 0.7,
        alpha: 0.5 + Math.random() * 0.5
    }));
    function draw() {
        ctx.clearRect(0, 0, w, h);
        for (const p of particles) {
            ctx.save();
            ctx.globalAlpha = p.alpha;
            ctx.shadowColor = color;
            ctx.shadowBlur = 12;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.restore();
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0 || p.x > w) p.vx *= -1;
            if (p.y < 0 || p.y > h) p.vy *= -1;
        }
        requestAnimationFrame(draw);
    }
    draw();
    return canvas;
} 