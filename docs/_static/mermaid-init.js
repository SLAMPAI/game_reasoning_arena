// Initialize Mermaid diagrams
document.addEventListener('DOMContentLoaded', function() {
    mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        securityLevel: 'loose',
        flowchart: {
            htmlLabels: true,
            curve: 'basis'
        }
    });
});
