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
1. Input a URL
2. Input prompt - The AI will use this to determine relevance of content
3. Output filename - Content will be saved in markdown format
4. The AI will:
   - Crawl only within the starting URL's folder structure
   - Extract content relevant to the prompt
   - Structure the output with source URLs and relevance scores
   
Keep logs or verbose output so users can track the crawler’s progress.

Project Scope
A Python-based utility that can be run from the command line (or integrated into other workflows).
Leverages Crawl4AI for JavaScript‐capable crawling, meaning sites with dynamic navigation or interactive elements can be captured.
Output is exported in a Markdown format, with optional chunking to avoid extremely large single files.
