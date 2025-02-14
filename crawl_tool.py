import asyncio
import re
from urllib.parse import urljoin, urlparse, urldefrag
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from typing import List, Optional, Set, Dict
import os

class ExtractedContent(BaseModel):
    content: str = Field(..., description="Extracted relevant content")
    relevance_score: float = Field(..., description="Score between 0-1 of relevance to prompt")
    source_url: str = Field(..., description="URL where content was found")

class WebCrawler:
    def __init__(self, verbose: bool = True):
        # Verify OpenRouter API key format
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or not api_key.startswith("sk-or-"):
            raise ValueError(
                "Invalid OpenRouter API key format\n"
                "1. Get your key from https://openrouter.ai/keys\n"
                "2. Update .env file with your actual key\n"
                "3. Valid keys start with 'sk-or-'"
            )

        # Initialize LLM extraction strategy with OpenRouter 
        self.llm_strategy = LLMExtractionStrategy(
            provider="openrouter_ai",
            model=os.getenv("OPENROUTER_MODEL", "deepseek-ai/deepseek-r1"),
            api_token=api_key,
            extraction_type="schema",
            schema=ExtractedContent.schema(),
            instruction=(
                "Extract documentation about IDL generation and integration with "
                "complete technical details, code samples and configuration. "
                "Include these 3 fields in JSON: "
                "content (full text with code), relevance_score (0-1), source_url"
            ),
            chunk_token_threshold=4096,
            base_url="https://openrouter.ai/api/v1",
            extra_args={
                "headers": {
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "Crawl4AI Research Tool",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-project",
                    "X-Title": "Anchor IDL Documentation Extraction"
                },
                "temperature": 0.3,
                "top_p": float(os.getenv("TOP_P", "0.9")),
                "response_format": {"type": "json_object"},
                "max_tokens": int(os.getenv("MAX_TOKENS", "32000")) 
            },
            verbose=verbose
        )
        
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            text_mode=False,
            verbose=verbose
        )
        
        self.md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.45,
                threshold_type="dynamic",
                min_word_threshold=5
            )
        )
        self.processed_urls = set()
        self.page_contents = {}

    def _get_wait_selector(self, url: str) -> str:
        parsed_url = urlparse(url)
        
        if parsed_url.netloc == 'docs.rs':
            return "css:.rustdoc"
        elif parsed_url.netloc == 'docs.crawl4ai.com':
            # Updated wait selector to a more reliable element
            return "css:body"
        elif 'readthedocs.io' in parsed_url.netloc:
            return "css:.document"
        
        return "css:body"

    def _create_link_filter(self, base_url: str) -> callable:
        """Creates a link filter constrained to the highest folder level."""
        parsed_base = urlparse(base_url)
        path_parts = parsed_base.path.strip('/').split('/')
        
        # Maintain structure up to last folder of base URL
        if len(path_parts) > 0 and '.' in path_parts[-1]:  # If has file extension
            base_depth = max(len(path_parts) - 1, 1)
        else:
            base_depth = len(path_parts)
            
        base_path = '/'.join(path_parts[:base_depth]) if base_depth > 0 else ''
        
        def link_filter(link: str) -> bool:
            # Remove anchor from URL
            link, _ = urldefrag(link)
            parsed_link = urlparse(link)
            
            # Must be same domain
            if parsed_link.netloc and parsed_link.netloc != parsed_base.netloc:
                return False
                
            # Must be under the same crate version path
            link_path = parsed_link.path.strip('/')
            if not link_path.startswith(base_path):
                return False
                
            return True
            
        return link_filter

    async def crawl(self, url: str, output_file: str, user_prompt: str = None, retry_count: int = 0) -> None:
        """Crawl the website and save content to markdown file."""
        # Validate URL format up front
        print(f"\nDEBUG: Validating URL: {url!r}")  # Show exact URL content
        if not url:
            raise ValueError("URL cannot be empty")
            
        url_pattern = r'^https?://([^\s/:?#]+\.)+[^\s/:?#]+(\/[^\s?#\/]+)*$'
        match = re.match(url_pattern, url, re.IGNORECASE)
        if not match:
            print(f"DEBUG: URL validation failed")
            print(f"DEBUG: URL parts:")
            print(f"- Protocol: {urlparse(url).scheme}")
            print(f"- Domain: {urlparse(url).netloc}")
            print(f"- Path: {urlparse(url).path}")
            raise ValueError(f"Invalid URL format: {url}\nPlease ensure it starts with http:// or https:// and has a valid domain structure")
        print(f"DEBUG: URL validation passed")

        # Store original URL for later use
        original_url = url
        
        # Remove any fragment identifier from the URL
        url, _ = urldefrag(url)
        base_url = url
        
        # Generate filename based on URL structure 
        parsed_url = urlparse(url)
        domain_parts = parsed_url.netloc.split('.')
        path_parts = parsed_url.path.strip('/').split('/')
        filename = f"{domain_parts[-2] if len(domain_parts) > 1 else domain_parts[0]}_{path_parts[-1] if path_parts else 'index'}.md"
        output_file = os.path.join('research', filename)
        os.makedirs('research', exist_ok=True)
        
        # Track all content
        all_content = []
        processed_urls = set()

        # Validate prompt or set default behavior
        if not user_prompt or len(user_prompt) < 10:
            print("\nWarning: No valid prompt provided. Extracting all content from landing page...")
            user_prompt = "Extract all meaningful content from the page, " \
                          "focusing on technical documentation, API references, " \
                          "and developer guides."

        # Configure the crawler with LLM strategy
        self.llm_strategy.instruction = user_prompt
        run_config = CrawlerRunConfig(
            extraction_strategy=self.llm_strategy,
            cache_mode=CacheMode.BYPASS,
            markdown_generator=self.md_generator,
            verbose=True,
            process_iframes=True,
            word_count_threshold=1,
            excluded_tags=[],
            remove_overlay_elements=True,
            wait_for=self._get_wait_selector(url),
            page_timeout=30000
        )

        # Create link filter
        link_filter = self._create_link_filter(base_url)

        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                # First process landing page
                try:
                    initial_result = await crawler.arun(url=url, config=run_config)
                except Exception as e:
                    print(f"\nRetrying landing page without wait condition...")
                    run_config = run_config.clone()
                    run_config.wait_for = None
                    initial_result = await crawler.arun(url=url, config=run_config)

                if not initial_result.success:
                    print(f"\n✗ Failed to process landing page: {url}")
                    if hasattr(initial_result, 'error_message'):
                        print(f"Error: {initial_result.error_message}")
                    return

                # Process landing page content
                if hasattr(initial_result, "html"):
                    # Use crawl4ai's cleaned markdown output directly
                    content = initial_result.markdown_v2.raw_markdown
                    if content:
                        print(f"\n✓ Successfully processed landing page: {url}")
                        all_content.append(f"# Documentation from {url}\n\n{content}\n")
                        processed_urls.add(url)

                # Get all links from the landing page
                urls_to_process = set()
                if hasattr(initial_result, "links"):
                    for link_data in initial_result.links.get("internal", []):
                        link_url = link_data.get("href")
                        if link_url:
                            # Remove any fragments and normalize
                            link_url, _ = urldefrag(link_url)
                            if link_url not in processed_urls and link_filter(link_url):
                                urls_to_process.add(link_url)

                # Process all discovered links
                print(f"\nFound {len(urls_to_process)} additional pages to process")
                for link_url in urls_to_process:
                    try:
                        print(f"\nProcessing: {link_url}")
                        result = await crawler.arun(url=link_url, config=run_config)
                        
                        if result.success and hasattr(result, "html"):
                            # Extract content using LLM with relevance scoring
                            try:
                                import json  # Add missing import
                                extracted = json.loads(result.extracted_content)
                                # Validate we got the expected fields
                                if not all('content' in item and 'relevance_score' in item and 'source_url' in item for item in extracted):
                                    raise ValueError("Missing required fields in LLM response")
                                    
                            except (json.JSONDecodeError, ValueError) as e:
                                print(f"Failed to parse LLM response: {str(e)}")
                                if retry_count < 3:
                                    print(f"Retrying... ({retry_count + 1}/3)")
                                    return await self.crawl(url, output_file, user_prompt, retry_count + 1)
                                else:
                                    print("Maximum retries exceeded. Falling back to full page scrape.")
                                    # Fallback to using cleaned HTML with source URL
                                    extracted = [{
                                        "content": result.cleaned_html or result.markdown_v2.raw_markdown,
                                        "relevance_score": 1.0,
                                        "source_url": link_url 
                                    }]
                                    print("Fallback to HTML content due to LLM parsing error")
                            for item in extracted:
                                content_item = ExtractedContent(**item)
                                if content_item.relevance_score >= 0.5:  # Threshold
                                    all_content.append(
                                        f"\n## Source: {content_item.source_url}\n"
                                        f"### Relevance Score: {content_item.relevance_score:.2f}\n"
                                        f"{content_item.content}\n"
                                    )
                                processed_urls.add(link_url)
                                print(f"✓ Successfully processed: {link_url}")
                            else:
                                print(f"✗ No content extracted from: {link_url}")
                        else:
                            print(f"✗ Failed to process: {link_url}")
                    except Exception as e:
                        print(f"✗ Error processing {link_url}: {str(e)}")

                # Save all content to file with automatic chunking
                if all_content:
                    try:
                        # Join all content
                        combined_content = '\n'.join(all_content)
                        lines = combined_content.splitlines()
                        total_lines = len(lines)
                        
                        # If content exceeds 3000 lines, create multiple chunks
                        if total_lines > 3000:
                            chunk_size = 3000
                            num_chunks = (total_lines + chunk_size - 1) // chunk_size
                            print(f"\nContent is {total_lines} lines long. Splitting into {num_chunks} chunks...")
                            
                            base_name = output_file.rsplit('.', 1)[0]
                            for i in range(num_chunks):
                                start_idx = i * chunk_size
                                end_idx = min((i + 1) * chunk_size, total_lines)
                                chunk_content = '\n'.join(lines[start_idx:end_idx])
                                
                                # Add chunk information at the top
                                chunk_header = f"# Chunk {i+1} of {num_chunks}\n\n"
                                chunk_file = f"{base_name}_chunk_{i+1}.md"
                                
                                with open(chunk_file, 'w', encoding='utf-8') as f:
                                    f.write(chunk_header + chunk_content)
                                print(f"✓ Created chunk {i+1}: {chunk_file} ({end_idx-start_idx} lines)")
                            
                            print(f"\n✓ Successfully split content from {len(processed_urls)} pages into {num_chunks} chunks")
                        else:
                            # Content is small enough for one file
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(combined_content)
                            print(f"\n✓ Successfully saved content from {len(processed_urls)} pages to {output_file} ({total_lines} lines)")
                    except Exception as e:
                        print(f"\n✗ Error saving to file: {str(e)}")
                else:
                    print("\n✗ No content was extracted")

        except Exception as e:
            print(f"\n✗ Error during crawling: {str(e)}")
            return

async def main():
    print("\nWeb Content Crawler")
    print("==================")
    
    url = input("\nEnter URL: ").strip()
    if not url:
        raise ValueError("URL cannot be empty")
        
    user_prompt = input("Enter research prompt (or leave empty for full extraction): ").strip()

    print(f"\nStarting crawl of: {url!r}")  # Debug - show exact URL including quotes
    crawler = WebCrawler(verbose=True)
    try:
        await crawler.crawl(url, "", user_prompt if len(user_prompt) > 10 else None)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Attempted URL: {url!r}")  # Debug - show exact URL that caused error
        raise

# Load environment variables first thing
load_dotenv()

if __name__ == "__main__":
    asyncio.run(main())
