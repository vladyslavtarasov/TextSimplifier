// Listens for messages to update the text input field in the popup with processed text.
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
    if (message.action === "updateTextInput") {
        document.getElementById('textInput').value = message.text;
    }
});

// Initializes slider controls for setting thresholds and binds events to update and store their values.
document.addEventListener('DOMContentLoaded', function() {
    const originalSlider = document.getElementById('originalSlider');
    const newSlider = document.getElementById('newSlider');
    const summarizationSlider = document.getElementById('summarizationSlider');

    const originalSliderValue = document.getElementById('slider1Value');
    const newSliderValue = document.getElementById('slider2Value');
    const summarizationSliderValue = document.getElementById('slider3Value');

    function updateSliderValues() {
        chrome.storage.local.set({
            originalSliderVal: originalSlider.value,
            newSliderVal: newSlider.value,
            summarizationSliderVal: summarizationSlider.value
        });
        originalSliderValue.textContent = originalSlider.value;
        newSliderValue.textContent = newSlider.value;
        summarizationSliderValue.textContent = summarizationSlider.value;
    }

    originalSlider.addEventListener('input', updateSliderValues);
    newSlider.addEventListener('input', updateSliderValues);
    summarizationSlider.addEventListener('input', updateSliderValues);

    chrome.storage.local.get(['originalSliderVal', 'newSliderVal', 'summarizationSliderVal'], function(result) {
        originalSlider.value = result.originalSliderVal || 0.3;
        newSlider.value = result.newSliderVal || 0.3;
        summarizationSlider.value = result.summarizationSliderVal || 0.5;
        originalSliderValue.textContent = result.originalSliderVal || 0.3;
        newSliderValue.textContent = result.newSliderVal || 0.3;
        summarizationSliderValue.textContent = result.summarizationSliderVal || 0.5;
    });
});

// Handles button clicks to simplify or summarize text, sending requests to the content script.
document.addEventListener('DOMContentLoaded', function() {
    const simplifyBtn = document.getElementById('simplifyTextBtn');
    const summariseTextBtn = document.getElementById('summariseTextBtn');
    const textInput = document.getElementById('textInput');

    simplifyBtn.addEventListener('click', function() {
        const textToSimplify = textInput.value;
        if (!textToSimplify.trim()) {
            return;
        }

        chrome.storage.local.get(['originalSliderVal', 'newSliderVal'], function(result) {
            const originalTH = parseFloat(result.originalSliderVal || 0.3);
            const newTH = parseFloat(result.newSliderVal || 0.3);

            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'simplifyText',
                    text: textToSimplify,
                    originalThreshold: originalTH,
                    newThreshold: newTH
                });
            });
        });
    });

    summariseTextBtn.addEventListener('click', function() {
        const textToSummarize = textInput.value;
        if (!textToSummarize.trim()) {
            return;
        }

        chrome.storage.local.get(['summarizationSliderVal'], function(result) {
            const coeff = parseFloat(result.summarizationSliderVal || 0.5);

            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'summarizeText',
                    text: textToSummarize,
                    summarize_coefficient: coeff
                });
            });
        });
    });
});
