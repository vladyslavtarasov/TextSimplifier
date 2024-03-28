# Text Simplify Extension

This Chrome extension provides a convenient way to simplify and summarize text directly within your browser. 

### Features

- __Text Simplification:__ Simplify complex sentences with easier words and phrases, making the content more accessible.  
- __Text Summarization:__ Get concise summaries of lengthy articles, preserving the core message and saving time.  
- __Customizable Settings:__ Adjust the difficulty level for simplification and set preferred summarization length.  
- __Context Menu Integration:__ Simply right-click to simplify or summarize selected text.  
- __Highlighting & Tooltips:__ See what's changed with highlighted text and tooltips providing additional context.  

### Installation

1. Clone this repository or download the ZIP file.  
2. Open Chrome and navigate to chrome://extensions/.  
3. Enable "Developer mode"  .  
4. Click "Load unpacked" and select the extension directory (containing the manifest.json file).  

### Running the Flask App

The extension needs the Flask backend to process text simplification and summarization. 
Before using the extension, ensure the Flask app is running:

1. Navigate to the backend directory where app.py is located. 
2. Create and activate a virtual environment.  
To create a virtual environment:  
```python3 -m venv venv```  
To activate the environment:  
Windows: ```venv\Scripts\activate```  
Linux/macOS: ```esource venv/bin/activate```  
3. Install the required packages.  
```pip install -r requirements.txt```  
4. Set the FLASK_APP environment variable.  
Windows: ```set FLASK_APP=app.py```  
Linux/macOS: ```export FLASK_APP=app.py```  
5. Run the Flask application.  
```flask run```

### Usage  

- __To Simplify Text:__ Select text on any web page, right-click, and choose "Simplify selected text" from the context menu.  
- __To Summarize Text:__ Select text, right-click, and select "Summarize selected text".  
- __Adjusting Settings:__ Click the extension icon in the toolbar to open the popup. Use the sliders to adjust simplification difficulty and summarization length.  