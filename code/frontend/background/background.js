// Creates context menu items for text simplification and summarization on extension installation.
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "simplifySelectedText",
        title: "Simplify selected text",
        contexts: ["selection"]
    });

    chrome.contextMenus.create({
        id: "summarizeSelectedText",
        title: "Summarise selected text",
        contexts: ["selection"]
    });
});

// Responds to context menu selections, triggering text simplification or summarization.
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "simplifySelectedText") {
        chrome.tabs.sendMessage(tab.id, { action: "simplifySelectedText" });
    }

    if (info.menuItemId === "summarizeSelectedText") {
        chrome.tabs.sendMessage(tab.id, { action: "summarizeSelectedText" });
    }
});