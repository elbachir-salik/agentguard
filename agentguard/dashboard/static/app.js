// AgentGuard Dashboard - minimal JS
document.addEventListener('DOMContentLoaded', () => {
    // Highlight active nav link
    const path = window.location.pathname;
    document.querySelectorAll('nav a').forEach(a => {
        if (a.getAttribute('href') === path) {
            a.style.color = '#e6edf3';
            a.style.fontWeight = '600';
        }
    });
});
