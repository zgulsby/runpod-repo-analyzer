// Store the analysis results
let analysisResults = null;

// DOM Elements
const form = document.getElementById('analyze-form');
const loadingSection = document.getElementById('loading');
const resultsSection = document.getElementById('results');
const errorSection = document.getElementById('error');
const errorText = document.getElementById('error-text');
const fileDisplay = document.getElementById('file-display');
const downloadButton = document.getElementById('download-all');

// Initialize syntax highlighting
hljs.highlightAll();

// Apply high contrast colors to all elements in file content
function forceHighContrast() {
    console.log('Forcing high contrast colors');
    
    // Add inline styles for maximum contrast
    document.querySelectorAll('.file-content *').forEach(el => {
        if (el.tagName === 'SPAN') {
            // Apply different colors based on class for syntax highlighting
            if (el.classList.contains('hljs-attr')) {
                el.style.color = '#66ccff';
            } else if (el.classList.contains('hljs-string')) {
                el.style.color = '#ffcc66';
            } else if (el.classList.contains('hljs-number')) {
                el.style.color = '#99ff99';
            } else if (el.classList.contains('hljs-literal')) {
                el.style.color = '#ff99cc';
            } else {
                el.style.color = '#ffffff';
            }
            el.style.textShadow = '0 0 2px rgba(255, 255, 255, 0.2)';
        } else {
            el.style.color = '#ffffff';
        }
    });
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', forceHighContrast);

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const repoUrl = document.getElementById('repo-url').value;
    
    // Show loading state
    loadingSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    
    try {
        // Send analysis request
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: repoUrl })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Parse response
        analysisResults = await response.json();
        console.log("Raw response:", analysisResults);
        
        // Debug log for JSON content
        console.log("Hub JSON type:", typeof analysisResults.files.hub_json);
        console.log("Tests JSON type:", typeof analysisResults.files.tests_json);
        
        // Update UI with results
        updateResults(analysisResults);
        
        // Show results
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
    } catch (error) {
        console.error('Error:', error);
        loadingSection.classList.add('hidden');
        errorSection.classList.remove('hidden');
        errorText.textContent = error.message;
    }
});

// Update results in the UI
function updateResults(results) {
    // Update repository info
    document.getElementById('repo-type').textContent = results.repository.type;
    document.getElementById('repo-confidence').textContent = 
        `${results.repository.confidence.toFixed(2)}%`;
    document.getElementById('repo-languages').textContent = 
        results.repository.languages.join(', ');
    document.getElementById('repo-dependencies').textContent = 
        results.repository.dependencies.join(', ');
    
    // Set active tab to hub.json
    const hubTab = document.querySelector('[data-tab="hub"]');
    hubTab.click();
}

// Handle file tab switching
document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        // Update active tab
        document.querySelectorAll('.tab-btn').forEach(btn => 
            btn.classList.remove('active'));
        button.classList.add('active');
        
        // Get file content
        const fileType = button.dataset.tab;
        let content = '';
        let language = '';
        
        switch (fileType) {
            case 'hub':
                try {
                    const hubJson = analysisResults.files.hub_json;
                    // If it's a string that looks like JSON, try to parse it
                    if (typeof hubJson === 'string' && (hubJson.trim().startsWith('{') || hubJson.trim().startsWith('['))) {
                        try {
                            const parsed = JSON.parse(hubJson);
                            content = JSON.stringify(parsed, null, 2);
                        } catch {
                            // If parsing fails, use the raw string
                            content = hubJson;
                        }
                    } else if (typeof hubJson === 'object') {
                        // If it's already an object, stringify it
                        content = JSON.stringify(hubJson, null, 2);
                    } else {
                        content = hubJson || 'No content available';
                    }
                } catch (e) {
                    console.error("Error handling hub JSON:", e);
                    content = "Error displaying JSON";
                }
                language = 'json';
                break;
            case 'tests':
                try {
                    const testsJson = analysisResults.files.tests_json;
                    // If it's a string that looks like JSON, try to parse it
                    if (typeof testsJson === 'string' && (testsJson.trim().startsWith('{') || testsJson.trim().startsWith('['))) {
                        try {
                            const parsed = JSON.parse(testsJson);
                            content = JSON.stringify(parsed, null, 2);
                        } catch {
                            // If parsing fails, use the raw string
                            content = testsJson;
                        }
                    } else if (typeof testsJson === 'object') {
                        // If it's already an object, stringify it
                        content = JSON.stringify(testsJson, null, 2);
                    } else {
                        content = testsJson || 'No content available';
                    }
                } catch (e) {
                    console.error("Error handling tests JSON:", e);
                    content = "Error displaying JSON";
                }
                language = 'json';
                break;
            case 'dockerfile':
                content = analysisResults.files.dockerfile || 'No content available';
                language = 'dockerfile';
                break;
            case 'handler':
                content = analysisResults.files.handler || 'No content available';
                language = 'python';
                break;
        }
        
        // Update file display with content
        fileDisplay.textContent = content;
        fileDisplay.className = `language-${language}`;
        
        // Apply syntax highlighting
        hljs.highlightElement(fileDisplay);
    });
});

// Handle file downloads
downloadButton.addEventListener('click', () => {
    if (!analysisResults) return;
    
    // Create zip file
    const zip = new JSZip();
    
    // Add files to zip
    if (analysisResults.files.hub_json) {
        zip.file('hub.json', analysisResults.files.hub_json);
    }
    if (analysisResults.files.tests_json) {
        zip.file('tests.json', analysisResults.files.tests_json);
    }
    if (analysisResults.files.dockerfile) {
        zip.file('Dockerfile', analysisResults.files.dockerfile);
    }
    if (analysisResults.files.handler) {
        zip.file('handler.py', analysisResults.files.handler);
    }
    
    // Generate and download zip
    zip.generateAsync({ type: 'blob' })
        .then(content => {
            const url = window.URL.createObjectURL(content);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'runpod-files.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        });
}); 