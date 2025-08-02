// Initialize Mermaid diagrams
document.addEventListener('DOMContentLoaded', function() {
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {
                htmlLabels: true,
                curve: 'basis'
            },
            themeVariables: {
                primaryColor: '#e1f5fe',
                primaryTextColor: '#000',
                primaryBorderColor: '#007acc',
                lineColor: '#007acc'
            }
        });

        // Re-render any mermaid diagrams
        mermaid.init(undefined, document.querySelectorAll('.mermaid'));
    }
});
