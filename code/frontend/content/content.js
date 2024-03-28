 // Loads and injects a modal dialog HTML into the current webpage asynchronously.
async function injectAndShowModal(message) {
    const url = chrome.runtime.getURL('content/content.html');
    try {
        const response = await fetch(url);
        const html = await response.text();
        const div = document.createElement('div');
        div.innerHTML = html;
        document.body.appendChild(div.firstChild);

        document.getElementById('wrongTextModalText').innerText = message;

        const modal = document.getElementById('wrongTextModal');
        modal.style.display = 'block';

        document.querySelector('.wrong-text-modal-close-button').addEventListener('click', () => {
            document.getElementById('wrongTextModal').style.display = 'none';
        });

        window.addEventListener('click', function(event) {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        });
    } catch (error) {
        console.error('Error injecting HTML:', error);
    }
}

// Adds listeners to Handle incoming messages for text simplification and summarization actions.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'simplifySelectedText') {
        if(isSelectionAllowed()) {
            console.log("Selected text is in the single tag!")
            simplifySelectedText()
        }
        else {
            console.log("Selected text is NOT in the single tag!")
            injectAndShowModal("Do not select anything but text.");
        }
    }
    if (message.action === 'summarizeSelectedText') {
        if(isSelectionAllowed()) {
            console.log("Selected text is in the single tag!")
            summarizeSelectedText()
        }
        else {
            console.log("Selected text is NOT in the single tag!")
            injectAndShowModal("Do not select anything but text.");
        }
    }
    if (message.action === 'simplifyText') {
        simplifyText(message.text, message.originalThreshold, message.newThreshold);
    }
    if (message.action === 'summarizeText') {
        summarizeText(message.text, message.summarize_coefficient);
    }
});

// Evaluates the current text selection to determine if it's valid for processing.
// It checks if the selection is within allowed tags and does not include non-text elements.
function isSelectionAllowed() {
    const selection = window.getSelection();
    if (!selection.rangeCount) return false;

    const range = selection.getRangeAt(0);
    const allNodes = getAllNodesInRange(range);

    for (const node of allNodes) {
        if (!isNodeValid(node)) {
            return false;
        }
    }

    return true;
}

// Retrieves all DOM nodes within the specified range of the current selection.
function getAllNodesInRange(range) {
    const nodes = [];
    const treeWalker = document.createTreeWalker(
        range.commonAncestorContainer,
        NodeFilter.SHOW_ALL, // Consider all nodes
        null
    );

    while (treeWalker.nextNode()) {
        const currentNode = treeWalker.currentNode;
        if (range.intersectsNode(currentNode)) {
            nodes.push(currentNode);
        }
    }

    return nodes;
}

// Determines if a DOM node is valid for text operations by checking its type and tag name.
function isNodeValid(node) {
    if (node.nodeType === Node.TEXT_NODE) node = node.parentNode;
    console.log(node)
    if (node.nodeType === Node.ELEMENT_NODE && isAllowedTag(node)) {
        return true;
    }

    return false;
}

// Checks if an HTML element's tag name is in the list of allowed tags for text operations.
function isAllowedTag(element) {
    return ['P', 'UL', 'OL', 'LI', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'SPAN', 'A', 'DIV', 'SUP', 'FIGCAPTION',
    'STRONG', 'B', 'I', 'SECTION', 'EM'].includes(element.tagName);
}

// Captures the user-selected text and requests its simplification from the backend.
function simplifySelectedText() {
    const selection = window.getSelection();
    if (selection.rangeCount === 0) {
        return;
    }

    const range = selection.getRangeAt(0);
    const selectedText = range.toString().trim();

    if (selectedText.length === 0) {
        return;
    }

    chrome.storage.local.get(['originalSliderVal', 'newSliderVal'], function(result) {
        const originalTH = parseFloat(result.originalSliderVal || 0.3);
        const newTH = parseFloat(result.newSliderVal || 0.3);

        simplify(selectedText, originalTH, newTH).then(json => {
            const simplifiedTextWrapper = document.createElement('span');
            simplifiedTextWrapper.textContent = json.simplified_text;

            range.deleteContents();
            range.insertNode(simplifiedTextWrapper);

            selection.removeAllRanges();

            processTextHighlighting(json.word_mapping, simplifiedTextWrapper, json.additional_synonyms);
        });
    });
}

// Captures the user-selected text and requests its summarizations from the backend.
function summarizeSelectedText() {
    const selection = window.getSelection();
    if (selection.rangeCount === 0) {
        return;
    }

    const range = selection.getRangeAt(0);
    const selectedText = range.toString().trim();

    if (selectedText.length === 0) {
        return;
    }

    chrome.storage.local.get(['summarizationSliderVal'], function(result) {
        const coeff = parseFloat(result.summarizationSliderVal || 0.5);

        summarize(selectedText, coeff).then(json => {
            const summarizedTextWrapper = document.createElement('span');
            summarizedTextWrapper.textContent = json.summarized_text;

            range.deleteContents();
            range.insertNode(summarizedTextWrapper);

            selection.removeAllRanges();
        });
    });
}

// Highlights words in the container element that were replaced during simplification.
function highlightReplacedWords(wordReplacements, container, additionalSynonyms) {
    Object.entries(wordReplacements).forEach(([originalWord, replacementWord]) => {
        const synonyms = additionalSynonyms[originalWord] || [];
        const tooltipText = `<strong>Original word: </strong>${originalWord}
                                <br><strong>Additional synonyms: </strong>${synonyms.slice(0, 5).join(', ') || 'None'}`;

        const regex = new RegExp(`\\b${replacementWord}\\b`, 'g');
        container.innerHTML = container.innerHTML.replace(regex, (match) => {
            return `<span class="highlighted-word" data-tooltip="${tooltipText}">${match}</span>`;
        });
    });
}

// Highlights difficult words that were not replaced during simplification.
function highlightDifficultWords(additionalSynonyms, container, wordReplacements) {
    Object.keys(additionalSynonyms).forEach(word => {
        if (!(word in wordReplacements)) {
            const hasSynonyms = additionalSynonyms[word].length > 0;
            let className, tooltipText;

            if (hasSynonyms) {
                className = 'difficult-word-with-synonyms';
                tooltipText = `<strong>Synonyms: </strong>${additionalSynonyms[word].slice(0, 5).join(', ') || 'None'}`;
            } else {
                className = 'difficult-word-no-synonyms';
                tooltipText = `No synonyms found for this difficult word`;
            }

            const regex = new RegExp(`\\b${word}\\b`, 'g');
            container.innerHTML = container.innerHTML.replace(regex, (match) => {
                return `<span class="${className}" data-tooltip="${tooltipText}">${match}</span>`;
            });
        }
    });
}

// Attaches mouse event listeners to show and hide tooltips.
function attachTooltipListeners(container) {
    container.querySelectorAll('span[data-tooltip]').forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

function processTextHighlighting(wordReplacements, container, additionalSynonyms) {
    highlightReplacedWords(wordReplacements, container, additionalSynonyms);
    highlightDifficultWords(additionalSynonyms, container, wordReplacements);
    attachTooltipListeners(container);
}

function showTooltip(event) {
    const target = event.target;
    const tooltipText = target.getAttribute('data-tooltip');

    let tooltip = target.tooltipElement;
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.innerHTML = tooltipText;
        target.tooltipElement = tooltip;
        document.body.appendChild(tooltip);
    }

    tooltip.style.display = 'block';

    const rect = target.getBoundingClientRect();
    tooltip.style.top = `${window.scrollY + rect.bottom}px`;
    tooltip.style.left = `${window.scrollX + rect.left}px`;

}

function hideTooltip(event) {
    const target = event.target;
    const tooltip = target.tooltipElement;
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

// Captures the user-inserted text and requests its simplification from the backend.
function simplifyText(text, originalThreshold, newThreshold) {
    simplify(text, originalThreshold, newThreshold).then(json => {
        const simplifiedTextElement = document.createElement('div');
        simplifiedTextElement.innerHTML = json.simplified_text;
        processTextHighlighting(json.word_mapping, simplifiedTextElement, json.additional_synonyms);
        chrome.runtime.sendMessage({ action: "updateTextInput", text: json.simplified_text });
    });
}

// Captures the user-inserted text and requests its summarizations from the backend.
function summarizeText(text, coeff) {
    summarize(text, coeff).then(json => {
        const summarizedTextElement = document.createElement('div');
        summarizedTextElement.innerHTML = json.summarized_text;
        chrome.runtime.sendMessage({ action: "updateTextInput", text: json.summarized_text });
    });
}

