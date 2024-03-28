// Sends a request to simplify text with specified original and new difficulty thresholds.
function simplify(text, originalTH, newTH) {
    return new Promise((resolve, reject) => {
        fetch('http://127.0.0.1:8080/simplify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text, originalThreshold: originalTH,  newThreshold: newTH})
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(json => {
            resolve(json);
        })
        .catch(error => {
            reject(error);
        });
    });
}

// Sends a request to summarize text with a specified summarization coefficient.
function summarize(text, coefficient) {
    return new Promise((resolve, reject) => {
        fetch('http://127.0.0.1:8080/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text, summarizeCoefficient: coefficient})
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(json => {
            resolve(json);
        })
        .catch(error => {
            reject(error);
        });
    });
}
