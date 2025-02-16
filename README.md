The crawl_tool.py script aims to automate the process of collecting targeted documentation from dynamic web pages and compiling all relevant content into a single Markdown file with the assistance of an AI, so that the relevent information can be utalized by an AI as context in application development or other purposes. 

It manages everything from following links across multiple pages, to removing extraneous HTML or script tags, to ensuring that JavaScript-driven elements are fully loaded before gathering text.

Key Objectives

Comprehensive Crawling

Automatically traverse all sublinks within a domain up until the last folder in the starting url. IE if the starting url is https://www.domain.com/stuff/stuff1/stuff2/index.html, the crawler will only follow links that include the url www.domain.com/stuff/stuff1/stuff2/ 

Use a programmable “link filter” so relevant links are followed, while external domains are ignored.

Intelligent Content Capture
Employ a headless Chromium browser to wait for page elements that only appear after JavaScript runs.
Clean up the HTML to produce usable Markdown text without script references or other clutter.
Single Gathered Output

Run scraped content by an AI agent and let them select if it is relevent or not to the prompt. Consolidate the relevent content from every visited page into one combined Markdown file.

Extensibility for Dynamic Sites
Integrate with Crawl4AI’s advanced features (JavaScript injection, waiting for specific events, multi-step flows) to handle “Load More” buttons or multi-page forms.

Ease of Use
1. Input a URL - Must be a valid web address starting with http:// or https://
2. Input prompt - This guides the AI in determining content relevance (e.g. "Documentation about API endpoints")
3. Output filename - Content will be saved in markdown format in the 'research' folder
4. The AI will:
   - Crawl only within the starting URL's folder structure
   - Use AI (via OpenRouter) to filter content based on relevance
   - Structure output with source URLs and 0-1 relevance scores  
   - Default to full extraction if no prompt provided
   
Keep logs or verbose output so users can track the crawler’s progress.

Project Scope
## Features

- Two-pass LLM analysis for comprehensive documentation research
- Automated URL filtering and deduplication
- Intelligent content extraction with markdown formatting
- Configurable AI provider support (OpenRouter, OpenAI, Anthropic)
- Cost tracking for LLM API calls
- Automatic chunking of large outputs
- Detailed logging of crawler decisions

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
