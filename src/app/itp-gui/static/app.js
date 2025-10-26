// ITP GUI - Frontend JavaScript

const API_BASE = '';

// DOM Elements
let initButton, resetButton, runTacticButton, toggleDebugButton, validateButton, killButton, killButtonProof;
let projectRootInput, filePathInput, lemmaNameInput, tacticInput;
let initPanel, proofInterface;
let initStatus, tacticStatus, validationStatus;
let currentLemmaName, currentLemmaStmt, goalsContainer, errorsContainer;
let tacticHistory, debugContent, debugInfo, errorMessages;
let validationResults, validationLeanCode, validationOutput, validationErrors, validationDebugTraces;

// Application State
let state = {
    initialized: false,
    projectRoot: '',
    filePath: '',
    lemmaName: ''
};

// Initialize DOM references
document.addEventListener('DOMContentLoaded', () => {
    // Buttons
    initButton = document.getElementById('initButton');
    resetButton = document.getElementById('resetButton');
    runTacticButton = document.getElementById('runTacticButton');
    toggleDebugButton = document.getElementById('toggleDebug');
    validateButton = document.getElementById('validateButton');
    killButton = document.getElementById('killButton');
    killButtonProof = document.getElementById('killButtonProof');

    // Inputs
    projectRootInput = document.getElementById('projectRoot');
    filePathInput = document.getElementById('filePath');
    lemmaNameInput = document.getElementById('lemmaName');
    tacticInput = document.getElementById('tacticInput');

    // Panels
    initPanel = document.getElementById('initPanel');
    proofInterface = document.getElementById('proofInterface');

    // Status
    initStatus = document.getElementById('initStatus');
    tacticStatus = document.getElementById('tacticStatus');
    validationStatus = document.getElementById('validationStatus');

    // Proof State
    currentLemmaName = document.getElementById('currentLemmaName');
    currentLemmaStmt = document.getElementById('currentLemmaStmt');
    goalsContainer = document.getElementById('goalsContainer');
    errorsContainer = document.getElementById('errorsContainer');
    errorMessages = document.getElementById('errorMessages');

    // History and Debug
    tacticHistory = document.getElementById('tacticHistory');
    debugContent = document.getElementById('debugContent');
    debugInfo = document.getElementById('debugInfo');

    // Validation Results
    validationResults = document.getElementById('validationResults');
    validationLeanCode = document.getElementById('validationLeanCode');
    validationOutput = document.getElementById('validationOutput');
    validationErrors = document.getElementById('validationErrors');
    validationDebugTraces = document.getElementById('validationDebugTraces');

    // Event Listeners
    initButton.addEventListener('click', initializeSession);
    resetButton.addEventListener('click', resetSession);
    runTacticButton.addEventListener('click', runTactic);
    toggleDebugButton.addEventListener('click', toggleDebug);
    validateButton.addEventListener('click', validateProof);
    killButton.addEventListener('click', killExecutor);
    killButtonProof.addEventListener('click', killExecutor);

    // Enter key in tactic input
    tacticInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            runTactic();
        }
    });

    // Load saved values from localStorage
    loadSavedInputs();

    // Tab switching
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-button')) {
            const tabName = e.target.getAttribute('data-tab');
            switchTab(tabName, e.target);
        }
    });
});

// Switch tabs
function switchTab(tabName, button) {
    // Remove active class from all buttons and panes
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => btn.classList.remove('active'));
    tabPanes.forEach(pane => pane.classList.remove('active'));

    // Add active class to selected button and pane
    button.classList.add('active');
    document.getElementById(tabName).classList.add('active');
}

// Save input values to localStorage
function saveInputs() {
    localStorage.setItem('itp_project_root', projectRootInput.value);
    localStorage.setItem('itp_file_path', filePathInput.value);
    localStorage.setItem('itp_lemma_name', lemmaNameInput.value);
}

// Load saved input values
function loadSavedInputs() {
    const savedProjectRoot = localStorage.getItem('itp_project_root');
    const savedFilePath = localStorage.getItem('itp_file_path');
    const savedLemmaName = localStorage.getItem('itp_lemma_name');

    if (savedProjectRoot) projectRootInput.value = savedProjectRoot;
    if (savedFilePath) filePathInput.value = savedFilePath;
    if (savedLemmaName) lemmaNameInput.value = savedLemmaName;
}

// Show status message
function showStatus(element, message, type) {
    element.textContent = message;
    element.className = 'status-message ' + type;
    element.style.display = 'block';
}

// Hide status message
function hideStatus(element) {
    element.style.display = 'none';
}

// Initialize Session
async function initializeSession() {
    const projectRoot = projectRootInput.value.trim();
    const filePath = filePathInput.value.trim();
    const lemmaName = lemmaNameInput.value.trim();

    if (!projectRoot || !filePath || !lemmaName) {
        showStatus(initStatus, 'Please fill in all fields', 'error');
        return;
    }

    showStatus(initStatus, 'Initializing...', 'info');
    initButton.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/api/initialize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                project_root: projectRoot,
                file_path: filePath,
                lemma_name: lemmaName
            })
        });

        const data = await response.json();

        if (data.success) {
            state.initialized = true;
            state.projectRoot = projectRoot;
            state.filePath = filePath;
            state.lemmaName = lemmaName;

            saveInputs();

            showStatus(initStatus, 'Session initialized successfully!', 'success');

            // Switch to proof interface
            setTimeout(() => {
                initPanel.style.display = 'none';
                proofInterface.style.display = 'block';
                updateProofState(data.state);
                updateDebugInfo(data.debug);
            }, 500);
        } else {
            showStatus(initStatus, 'Error: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus(initStatus, 'Network error: ' + error.message, 'error');
    } finally {
        initButton.disabled = false;
    }
}

// Kill Executor
async function killExecutor() {
    if (!confirm('Are you sure you want to kill the executor? This will terminate the REPL process and clean up all resources.')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/kill`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            state.initialized = false;
            proofInterface.style.display = 'none';
            initPanel.style.display = 'block';
            tacticHistory.innerHTML = '<p class="empty-message">No tactics executed yet.</p>';
            showStatus(initStatus, data.message, 'success');
            setTimeout(() => hideStatus(initStatus), 3000);
        } else {
            showStatus(initStatus, 'Error killing executor: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus(initStatus, 'Error killing executor: ' + error.message, 'error');
    }
}

// Reset Session
async function resetSession() {
    if (!confirm('Are you sure you want to reset the session?')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/reset`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            state.initialized = false;
            proofInterface.style.display = 'none';
            initPanel.style.display = 'block';
            tacticHistory.innerHTML = '<p class="empty-message">No tactics executed yet.</p>';
            hideStatus(initStatus);
        }
    } catch (error) {
        showStatus(initStatus, 'Error resetting: ' + error.message, 'error');
    }
}

// Run Tactic
async function runTactic() {
    // Don't trim - preserve exact whitespace including tabs/spaces for Lean formatting
    const tactic = tacticInput.value;

    // Only check if it's completely empty (no trimming)
    if (!tactic || tactic.trim().length === 0) {
        showStatus(tacticStatus, 'Please enter a tactic', 'error');
        return;
    }

    showStatus(tacticStatus, 'Running tactic...', 'info');
    runTacticButton.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/api/run_tactic`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                tactic: tactic
            })
        });

        const data = await response.json();

        if (data.success) {
            showStatus(tacticStatus, 'Tactic executed!', 'success');
            tacticInput.value = '';
            updateProofState(data.state);
            updateDebugInfo(data.debug);

            setTimeout(() => hideStatus(tacticStatus), 2000);
        } else {
            showStatus(tacticStatus, 'Error: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus(tacticStatus, 'Network error: ' + error.message, 'error');
    } finally {
        runTacticButton.disabled = false;
        tacticInput.focus();
    }
}

// Update Proof State
function updateProofState(proofState) {
    // Update lemma info
    currentLemmaName.textContent = proofState.lemma_name || 'Unknown';

    // Update lemma statement with syntax highlighting
    const lemmaStmt = proofState.lemma_stmt || 'No statement available';
    currentLemmaStmt.innerHTML = '';
    const codeElement = document.createElement('code');
    codeElement.className = 'language-haskell';
    codeElement.textContent = lemmaStmt;
    currentLemmaStmt.appendChild(codeElement);

    // Apply syntax highlighting
    if (typeof Prism !== 'undefined') {
        Prism.highlightElement(codeElement);
    }

    // Update goals
    goalsContainer.innerHTML = '';
    if (proofState.goals && proofState.goals.length > 0) {
        proofState.goals.forEach((goal, index) => {
            const goalDiv = document.createElement('div');
            goalDiv.className = 'goal-item';

            const goalTitle = document.createElement('h4');
            goalTitle.textContent = `Goal ${index + 1}:`;
            goalDiv.appendChild(goalTitle);

            if (goal.hypotheses && goal.hypotheses.length > 0) {
                const hypothesesDiv = document.createElement('div');
                hypothesesDiv.className = 'hypotheses';

                const hypTitle = document.createElement('strong');
                hypTitle.textContent = 'Hypotheses:';
                hypothesesDiv.appendChild(hypTitle);

                goal.hypotheses.forEach(hyp => {
                    const hypDiv = document.createElement('div');
                    hypDiv.className = 'hypothesis';
                    hypDiv.textContent = hyp;
                    hypothesesDiv.appendChild(hypDiv);
                });

                goalDiv.appendChild(hypothesesDiv);
            }

            const conclusionDiv = document.createElement('div');
            conclusionDiv.className = 'goal-conclusion';
            const conclusionTitle = document.createElement('strong');
            conclusionTitle.textContent = '⊢ Goal:';
            conclusionDiv.appendChild(conclusionTitle);

            const goalPre = document.createElement('pre');
            goalPre.textContent = goal.goal || 'No goal';
            conclusionDiv.appendChild(goalPre);

            goalDiv.appendChild(conclusionDiv);
            goalsContainer.appendChild(goalDiv);
        });
    } else {
        const noGoals = document.createElement('p');
        noGoals.className = 'empty-message';
        noGoals.textContent = proofState.is_in_proof_mode ? 'No goals remaining! Proof complete!' : 'Not in proof mode';
        goalsContainer.appendChild(noGoals);
    }

    // Update error messages
    if (proofState.error_messages && proofState.error_messages.length > 0) {
        errorMessages.style.display = 'block';
        errorsContainer.innerHTML = '';
        proofState.error_messages.forEach(error => {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-item';
            errorDiv.textContent = error;
            errorsContainer.appendChild(errorDiv);
        });
    } else {
        errorMessages.style.display = 'none';
    }

    // Update history
    updateHistory(proofState.history);
}

// Update Tactic History
function updateHistory(history) {
    if (!history || history.length === 0) {
        tacticHistory.innerHTML = '<p class="empty-message">No tactics executed yet.</p>';
        return;
    }

    tacticHistory.innerHTML = '';
    history.forEach((item, index) => {
        const historyDiv = document.createElement('div');
        historyDiv.className = 'tactic-item';
        if (item.errors && item.errors.length > 0) {
            historyDiv.classList.add('error');
        }

        const tacticCode = document.createElement('div');
        tacticCode.className = 'tactic-code';
        tacticCode.textContent = `${index + 1}. ${item.tactic}`;
        historyDiv.appendChild(tacticCode);

        const tacticMeta = document.createElement('div');
        tacticMeta.className = 'tactic-meta';
        tacticMeta.textContent = `Line: ${item.line_num} | Success: ${item.success}`;
        if (item.errors && item.errors.length > 0) {
            tacticMeta.textContent += ' | Errors: ' + item.errors.join(', ');
        }
        historyDiv.appendChild(tacticMeta);

        tacticHistory.appendChild(historyDiv);
    });

    // Scroll to bottom
    tacticHistory.scrollTop = tacticHistory.scrollHeight;
}

// Update Debug Info
function updateDebugInfo(debug) {
    if (!debug) return;

    const formatted = JSON.stringify(debug, null, 2);
    debugContent.textContent = formatted;
}

// Toggle Debug Panel
function toggleDebug() {
    if (debugInfo.style.display === 'none') {
        debugInfo.style.display = 'block';
    } else {
        debugInfo.style.display = 'none';
    }
}

// Validate Proof
async function validateProof() {
    showStatus(validationStatus, 'Validating proof with lake lean...', 'info');
    validateButton.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/api/validate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                timeout_sec: 30
            })
        });

        const data = await response.json();

        if (data.success) {
            const validation = data.validation;

            if (validation.success) {
                showStatus(validationStatus, '✓ Proof is complete and valid!', 'success');
            } else if (!validation.compilation_ok) {
                showStatus(validationStatus, '✗ Compilation failed: ' + validation.error_message, 'error');
            } else if (validation.has_sorries) {
                showStatus(validationStatus, '✗ Proof has unsolved goals (sorries)', 'error');
            } else {
                showStatus(validationStatus, '✗ ' + validation.error_message, 'error');
            }

            // Display validation results
            displayValidationResults(validation);

            // Auto-hide success message after 3 seconds
            if (validation.success) {
                setTimeout(() => hideStatus(validationStatus), 3000);
            }
        } else {
            showStatus(validationStatus, 'Error: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus(validationStatus, 'Network error: ' + error.message, 'error');
    } finally {
        validateButton.disabled = false;
    }
}

// Display Validation Results in Tabs
function displayValidationResults(validation) {
    // Show the validation results panel
    validationResults.style.display = 'block';

    // Populate Lean Code tab with syntax highlighting
    const leanCode = validation.lean_code || 'No code available';
    validationLeanCode.innerHTML = '';
    const codeElement = document.createElement('code');
    codeElement.className = 'language-haskell';  // Using Haskell as closest syntax
    codeElement.textContent = leanCode;
    const preElement = document.createElement('pre');
    preElement.className = 'line-numbers';
    preElement.appendChild(codeElement);
    validationLeanCode.appendChild(preElement);

    // Apply syntax highlighting
    if (typeof Prism !== 'undefined') {
        Prism.highlightElement(codeElement);
    }

    // Populate Output tab
    const outputText = `File: ${validation.temp_filename || 'N/A'}
Full Path: ${validation.temp_file_path || 'N/A'}
File Kept: ${validation.temp_file_kept ? 'Yes' : 'No'}
Return Code: ${validation.return_code}
Success: ${validation.success}
Compilation OK: ${validation.compilation_ok}
Has Sorries: ${validation.has_sorries}
Error Message: ${validation.error_message}

${validation.full_output || (validation.stdout + '\n' + validation.stderr)}`;
    validationOutput.textContent = outputText;

    // Populate Errors tab
    if (validation.errors && validation.errors.length > 0) {
        const errorsText = validation.errors.map((err, idx) =>
            `[${idx + 1}] ${err.file}:${err.line}:${err.column} (${err.severity})\n${err.message}`
        ).join('\n\n');
        validationErrors.textContent = errorsText;
    } else {
        validationErrors.textContent = 'No errors found';
    }

    // Populate Debug Traces tab
    if (validation.debug_traces && validation.debug_traces.length > 0) {
        const tracesText = validation.debug_traces.map((trace, idx) =>
            `[${idx + 1}] ${trace}`
        ).join('\n');
        validationDebugTraces.textContent = tracesText;
    } else {
        validationDebugTraces.textContent = 'No debug traces available';
    }

    // Add proof tactics info to output if available
    if (validation.possible_proof_tactics) {
        const currentOutput = validationOutput.textContent;
        validationOutput.textContent = currentOutput + '\n\n=== PROOF TACTICS (from possible_proof_tactics) ===\n' + validation.possible_proof_tactics;
    }
}
