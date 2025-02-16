# Crawl Tool: A Simple Overview

The `crawl_tool.py` script automatically collects and compiles documentation from dynamic websites into a single Markdown file. An AI helps decide which content is most useful for your needs, such as app development or providing context for another AI. This is extremily useful for quickly fetching and preparing context for AI agents in performing tasks.

---

## What Does It Do?

1. **Automatic Website Crawling**  
   - **Deep Link Navigation:**  
     Starts at your chosen URL and only follows links within that specific folder.  
     *Example:* If you start at `https://www.domain.com/stuff/stuff1/stuff2/index.html`, the tool will crawl only pages under `/stuff/stuff1/stuff2/`.
   - **Smart Filtering:**  
     Only follows links from the same website, ignoring external sites.

2. **Intelligent Content Extraction**  
   - **JavaScript-Aware:**  
     Uses a headless Chromium browser to wait for all dynamic (JavaScript-driven) content to load.
   - **Clean Output:**  
     Strips away extra HTML and script tags to provide neat Markdown text.

3. **AI-Powered Relevance Filtering**  
   - **First Pass:**  
     The AI reviews the landing page to pick out key documentation links based on your prompt.
   - **Second Pass:**  
     After gathering initial content, the AI conducts a second analysis to find even more useful links and details.
   - **Custom Prompts:**  
     Supply your own research prompt (e.g., "Documentation about API endpoints") or use a default prompt.

4. **Consolidated Output**  
   - **Single Markdown File:**  
     All relevant content is merged into one Markdown file. If the file is too large, it is split into smaller, manageable chunks.
   - **Source Information:**  
     Each section includes the source URL and a relevance score, helping you track where the information came from.

5. **Dynamic and Extensible**  
   - **Handles Dynamic Sites:**  
     Capable of managing “Load More” buttons, multi-page forms, and other dynamic elements.
   - **Multiple AI Providers:**  
     Supports various AI backends (OpenRouter, OpenAI, Anthropic) for flexibility.
   - **Cost and Log Tracking:**  
     Keeps detailed logs of each step and tracks the cost of AI API calls.

---

## How to Use the Tool

1. **Enter a URL:**  
   - The URL must start with `http://` or `https://`.

2. **Enter a Research Prompt:**  
   - This prompt guides the AI in identifying the most relevant content.
   - If left empty, a default prompt (set in the environment variables) is used.

3. **Specify an Output Filename:**  
   - The gathered content is saved as a Markdown file in the `research` folder.

4. **The Process:**  
   - **Validation:** The tool first checks that the URL is


## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crawl-tool.git
cd crawl-tool

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.template .env
```

1. Edit the `.env` file:
   - Uncomment your preferred LLM provider
   - Add your API key
   - Optionally uncomment/update MAIN_RESEARCH_PROMPT

## Usage

```bash
python3 crawl_tool.py
```

1. Enter a valid documentation URL when prompted
2. Enter research prompt OR press enter to use default
3. The tool will:
   - Perform initial analysis of landing page
   - Crawl first-pass recommended links
   - Perform second-pass analysis of aggregated content
   - Crawl additional recommended links
   - Save consolidated markdown to research/ folder

## Configuration Tips

- To skip the interactive prompt, set MAIN_RESEARCH_PROMPT in .env
- Adjust MAX_TOKENS (default 32000) for larger documentation sets
- Lower TOP_P (0.0-1.0) for more focused responses
- Logs are saved to logs/ folder for debugging


## License

MIT License

Copyright (c) 2025 Noah Severs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
