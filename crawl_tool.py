import asyncio
import json
import re
from urllib.parse import urljoin, urlparse, urldefrag
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from pydantic import BaseModel, Field
from typing import List
import os
import litellm

def strip_markdown_links(text: str) -> str:
    """
    Remove markdown hyperlink markup but keep any link text.
    For example:
      - "[Link Text](http://example.com "tooltip")" becomes "Link Text"
      - "String[](https://docs.rs/...)" becomes "String"
    """
    # This regex captures the text between the square brackets (which may be empty)
    # and then drops the entire parenthesized URL/tooltip.
    return re.sub(r'\[([^\]]*)\]\([^)]*\)', lambda m: m.group(1), text)


# Define our extracted content schema
class ExtractedContent(BaseModel):
    relevant_urls: List[str] = Field(
        default_factory=list,
        description="List of URLs relevant to the research prompt (first pass)"
    )
    second_relevant_urls: List[str] = Field(
        default_factory=list,
        description="Additional URLs found in scraped content (second pass)"
    )
    relevance_reasons: List[str] = Field(
        default_factory=list,
        description="Brief reason for each URL's relevance across both passes"
    )
    main_topic: str = Field(
        default="Unknown topic - LLM failed to identify main theme",
        description="Primary topic identified from landing page"
    )

class WebCrawler:
    def __init__(self, verbose: bool = True):
        self.total_cost = 0.0
        # Define provider order
        providers = ["openrouter", "openai", "anthropic"]
        keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY")
        }
        
        # Check for an explicit provider
        explicit_provider = os.getenv("LLM_PROVIDER")
        if explicit_provider:
            explicit_provider = explicit_provider.lower().strip()
            if explicit_provider not in providers:
                raise ValueError(f"LLM_PROVIDER value '{explicit_provider}' is not one of {providers}.")
            api_key = keys.get(explicit_provider)
            if not api_key or not api_key.strip():
                raise ValueError(f"LLM_PROVIDER is set to '{explicit_provider}' but its corresponding key is not set.")
            provider_name = explicit_provider
        else:
            # Select the first available key in priority order
            provider_name, api_key = None, None
            for p in providers:
                key_val = keys.get(p)
                if key_val and key_val.strip():
                    provider_name, api_key = p, key_val
                    break
            if not provider_name:
                raise ValueError("No LLM key set in environment. Please set one of OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY.")
        
        # Prepare extra_args WITHOUT any callbacks (to ensure JSON serializability)
        extra_args = {
            "headers": {
                "HTTP-Referer": "https://github.com/nsevers/crawl_tool",
                "X-Title": "Smart Documentation Extraction",
                "Content-Type": "application/json",
                "X-API-Key": api_key
            },
            "temperature": 0.3,
            "top_p": float(os.getenv("TOP_P", "0.9")),
            "response_format": {"type": "json_object"},
            "max_tokens": int(os.getenv("MAX_TOKENS", "32000"))
        }
        # Initialize the LLM extraction strategy (its instruction will be updated later if a prompt is provided)
        self.llm_strategy = LLMExtractionStrategy(
            provider=(
                os.getenv("OPENAI_MODEL", "openai/gpt-3.5-turbo") if provider_name == "openai"
                else os.getenv("ANTHROPIC_MODEL", "anthropic/claude-v2") if provider_name == "anthropic"
                else os.getenv("OPENROUTER_MODEL", "openrouter/deepseek/deepseek-r1")
            ),
            api_token=api_key,
            extraction_type="schema",
            schema=ExtractedContent.model_json_schema(),
            instruction=(
                "Analyze the landing page and identify relevant URLs for documentation research. "
                "Be discriminating, only follow URLs that are directly related to the main topic and will be useful. "
                "Return JSON with: 1) relevant_urls 2) brief relevance_reasons 3) main_topic. "
                "Only include URLs within the same documentation section/subfolder. "
                "Aim to identify 5-20 high-value links that directly relate to the research prompt."
            ),
            api_base="https://openrouter.ai/api/v1",
            extra_args=extra_args,
            verbose=verbose
        )
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=verbose
        )
        self.page_generator = DefaultMarkdownGenerator()
        
        self.llm_generator = DefaultMarkdownGenerator(
            options={"ignore_images": True, "skip_internal_links": True})
        
        self.processed_urls = set()

    async def crawl(self, url: str, user_prompt: str = None) -> None:
        # --- URL Validation ---
        print(f"\nDEBUG: Validating URL: {url!r}")
        if not url:
            raise ValueError("URL cannot be empty")
        url_pattern = r'^https?://([^\s/:?#]+\.)+[^\s/:?#]+(\/[^\s?#\/]+)*$'
        if not re.match(url_pattern, url, re.IGNORECASE):
            raise ValueError(f"Invalid URL format: {url}")
        print("DEBUG: URL validation passed")

        # --- Auto-generate output filename based on URL ---
        parsed_url = urlparse(url)
        domain_parts = parsed_url.netloc.split('.')
        path_parts = parsed_url.path.strip('/').split('/')
        filename = f"{domain_parts[-2] if len(domain_parts) > 1 else domain_parts[0]}_{(path_parts[-1] if path_parts and path_parts[-1] else 'index')}.md"
        output_file = os.path.join('research', filename)
        os.makedirs('research', exist_ok=True)

        # --- Validate prompt or set default ---
        default_prompt = os.getenv(
            "DEFAULT_RESEARCH_PROMPT",
            "Extract all meaningful content from the page, focusing on technical documentation, API references, and developer guides."
        )
        if not user_prompt or len(user_prompt) < 10:
            print("\nWarning: No valid prompt provided. Using default prompt from .env.")
            user_prompt = default_prompt
        else:
            print("Using provided research prompt.")

        # Update the LLM extraction strategy with the user prompt
        self.llm_strategy.instruction = user_prompt

        base_url = url

        # --- Configure run configs ---
        llm_run_config = CrawlerRunConfig(
            extraction_strategy=self.llm_strategy,
            cache_mode=CacheMode.BYPASS,
            markdown_generator=self.llm_generator,
            verbose=True,
            process_iframes=True,
            word_count_threshold=1,
            wait_for="css:body",
            page_timeout=30000
        )

        schema = {
            "name": "Documentation Research",
            "baseSelector": "body",
            "fields": [
                {"name": "content", "selector": "body", "type": "markdown"}
            ]
        }
        non_llm_strategy = JsonCssExtractionStrategy(schema, verbose=True)
        non_llm_config = CrawlerRunConfig(
            extraction_strategy=non_llm_strategy,
            cache_mode=CacheMode.BYPASS,
            markdown_generator=self.page_generator,
            verbose=False,
            process_iframes=True,
            word_count_threshold=1,
            wait_for="css:body",
            page_timeout=15000,
            excluded_tags=["nav", "aside", "header", "footer"],
            # Exclude side menu selectors common across pages.
            excluded_selector="#ads, .tracker, .sidebar, nav.sidebar, .nav-container, nav.mobile-topbar"
        )

        def link_filter(link: str) -> bool:
            return urlparse(link).netloc == urlparse(base_url).netloc

        urls_to_process = set()
        all_content = []

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            # --- Process landing page ---
            try:
                initial_result = await crawler.arun(url=url, config=llm_run_config)
            except Exception as e:
                print(f"Error processing landing page: {str(e)}")
                return

            if not initial_result.success:
                print(f"Failed to process landing page: {url}")
                return

            # --- Cost tracking ---
            try:
                raw_llm = getattr(initial_result, "llm_response", None) or getattr(initial_result, "raw_response", None)
                if raw_llm is None:
                    raise ValueError("No raw LLM response available for cost tracking.")
                cost = litellm.completion_cost(completion_response=raw_llm)
                self.total_cost += cost
                print(f"\nLLM API Call Cost: ${cost:.6f} | Total: ${self.total_cost:.6f}")
            except Exception as e:
                print(f"LLM cost tracking failed: {str(e)}")

            # --- Process landing page content ---
            try:
                if not hasattr(initial_result, "html"):
                    raise ValueError("No HTML content found in landing page result.")
                # Use raw_markdown if available; otherwise fall back to HTML.
                content = (initial_result.markdown_v2.raw_markdown
                           if hasattr(initial_result, "markdown_v2") and initial_result.markdown_v2.raw_markdown
                           else initial_result.html)
                main_topic = "N/A"
                link_summary = "No recommended links found."
                content_data = initial_result.extracted_content
                if not content_data:
                    raise ValueError("Empty LLM response for extracted content.")
                if isinstance(content_data, str):
                    content_data = json.loads(content_data)
                if isinstance(content_data, list) and len(content_data) > 0:
                    content_data = content_data[0]
                content_item = ExtractedContent.model_validate(content_data)
                main_topic = content_item.main_topic
                if content_item.relevant_urls and content_item.relevance_reasons:
                    link_summary = "\n".join(
                        [f"- {u}: {r}" for u, r in zip(content_item.relevant_urls, content_item.relevance_reasons)]
                    )
            except Exception as e:
                print(f"Error parsing LLM landing page data: {str(e)}")
                print("Aborting crawl due to invalid LLM response from the landing page.")
                return
            header = (f"# Landing Page Analysis\nURL: {url}\n"
                      f"### Research Prompt: {user_prompt}\n"
                      f"Main Topic: {main_topic}\n\n"
                      f"### Recommended Links Summary\n{link_summary}\n\n"
                      f"### Second Pass Summary After Review\nWe decided to crawl these additional links:\n")
            all_content.append(header + content)
            self.processed_urls.add(url)
            

            # --- Extract recommended URLs from LLM response ---
            try:
                content_data = initial_result.extracted_content
                if isinstance(content_data, str):
                    content_data = json.loads(content_data)
                if isinstance(content_data, list) and len(content_data) > 0:
                    content_data = content_data[0]
                content_item = ExtractedContent.model_validate(content_data)
            except Exception as e:
                print(f"Error parsing LLM response: {str(e)}")
                content_item = ExtractedContent(
                    relevant_urls=[url],
                    relevance_reasons=["Fallback due to parsing error"],
                    main_topic="Extraction error"
                )

            for rec_url in content_item.relevant_urls:
                clean_url = urljoin(base_url, rec_url)
                # Remove any fragment identifier (anchor)
                clean_url, _ = urldefrag(clean_url)
                # Remove any stray angle-bracketed substrings
                clean_url = re.sub(r'<[^>]*>', '', clean_url)
                if clean_url not in self.processed_urls and link_filter(clean_url):
                    urls_to_process.add(clean_url)
            if len(urls_to_process) == 0:
                print("Warning: The LLM Failed to recommend any relevant URLs.")

            # First pass processing
            print(f"\nProcessing {len(urls_to_process)} first-pass LLM-recommended pages")
            first_pass_content = []
            for link_url in urls_to_process:
                try:
                    result = await crawler.arun(url=link_url, config=non_llm_config)
                    if not result.success:
                        raise ValueError("Crawl unsuccessful")
                    if not hasattr(result, "html"):
                        raise ValueError("No HTML content found")
                    # Process the content using the built-in fit_markdown function
                    raw_content = result.markdown_v2.raw_markdown
                    # Post-process the content to remove hyperlink markup while keeping the link text and tooltips.
                    content = strip_markdown_links(raw_content)

                    page_header = f"## Page Content ----------  URL: {link_url} ---------- \n\n"
                    page_content = page_header + content
                    all_content.append(page_content)
                    first_pass_content.append(page_content)  # Save for second pass analysis
                    self.processed_urls.add(link_url)
                    print(f"Processed: {link_url}")
                except Exception as e:
                    print(f"Error processing {link_url}: {str(e)}")
            

            # --- Second LLM Pass ---
            # Set up logger for second pass
            os.makedirs('logs', exist_ok=True)  # Create logs directory first
            import logging
            # Configure root logger to only write to files
            logging.basicConfig(
                level=logging.WARNING,
                handlers=[logging.NullHandler()]
            )
            # Configure our specific logger
            logger = logging.getLogger('SecondPass')
            logger.setLevel(logging.INFO)
            logger.propagate = False  # Prevent propagation to root logger
            # Clear any existing handlers
            logger.handlers = []
            handler = logging.FileHandler('logs/second_pass_analysis.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)

            logger.info("\nStarting second LLM analysis pass on scraped content...")
            second_pass_urls = set()
            
            try:
                # Update LLM instruction for second pass
                self.llm_strategy.instruction = (
                    f"Review the aggregated documentation content and identify 5-20 additional URLs "
                    f"that are critical for understanding '{main_topic}'. Focus on finding:\n"
                    f"- Deep links to specific API references\n"
                    f"- Advanced usage examples\n"
                    f"- Configuration guides\n"
                    f"- Integration documentation\n"
                    f"Original research prompt: {user_prompt}\n"
                    f"Return JSON with second_relevant_urls and updated relevance_reasons."
                )
                
                # Create temporary config for second pass analysis
                second_llm_config = CrawlerRunConfig(
                    extraction_strategy=self.llm_strategy,
                    cache_mode=CacheMode.BYPASS,
                    verbose=True
                )
                
                # Create synthetic URL with aggregated content
                synthetic_content = "\n".join(first_pass_content)
                # Log the synthetic content being sent to LLM
                with open('logs/second_pass_input.log', 'w') as f:
                    f.write(synthetic_content)
                synthetic_url = f"raw://{synthetic_content}"
                
                # Run second LLM analysis
                second_llm_result = await crawler.arun(
                    url=synthetic_url,
                    config=second_llm_config
                )
                
                # Track second pass cost
                try:
                    raw_llm = getattr(second_llm_result, "llm_response", None) or getattr(second_llm_result, "raw_response", None)
                    if raw_llm:
                        cost = litellm.completion_cost(completion_response=raw_llm)
                        self.total_cost += cost
                        print(f"\nSecond Pass LLM Cost: ${cost:.6f} | Total: ${self.total_cost:.6f}")
                except Exception as e:
                    print(f"Second pass cost tracking failed: {str(e)}")

                # Configure file logger for LLM interactions
                os.makedirs('logs', exist_ok=True)
                import logging
                logger = logging.getLogger('LLMInteraction')
                logger.setLevel(logging.DEBUG)
                handler = logging.FileHandler('logs/crawl_interactions.log')
                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(handler)

                # Process second pass recommendations with better error handling
                if second_llm_result.extracted_content:
                    try:
                        # Log the raw LLM response
                        logger.debug("Second pass LLM response:\n%s", second_llm_result.extracted_content)
                        
                        content_data = json.loads(second_llm_result.extracted_content)
                        
                        # Handle case where response might be a list
                        if isinstance(content_data, list):
                            content_data = content_data[0]  # Take first item if list
                            
                        content_item = ExtractedContent.model_validate(content_data)
                        
                        # Process second relevant URLs
                        for rec_url in content_item.second_relevant_urls:
                            clean_url = urljoin(base_url, rec_url)
                            clean_url, _ = urldefrag(clean_url)
                            clean_url = re.sub(r'<[^>]*>', '', clean_url)
                            if clean_url not in self.processed_urls and link_filter(clean_url):
                                second_pass_urls.add(clean_url)
                    except Exception as e:
                        print(f"Error processing second pass recommendations: {str(e)}")
                            
                    # Add second pass links to header
                    if second_pass_urls:
                        second_pass_summary = "\n".join([f"- {url}" for url in second_pass_urls])
                        all_content[0] += f"{second_pass_summary}\n\n"
                    
                    print(f"Found {len(second_pass_urls)} second-pass recommendations")
                    
                    # Process second pass URLs
                    if second_pass_urls:
                        print(f"\nProcessing {len(second_pass_urls)} second-pass LLM-recommended pages")
                        for link_url in second_pass_urls:
                            try:
                                result = await crawler.arun(url=link_url, config=non_llm_config)
                                if not result.success:
                                    continue
                                raw_content = result.markdown_v2.raw_markdown
                                content = strip_markdown_links(raw_content)
                                page_header = f"## Page Content ----------  URL: {link_url} ---------- \n\n"
                                all_content.append(page_header + content)
                                self.processed_urls.add(link_url)
                                print(f"Processed: {link_url}")
                            except Exception as e:
                                print(f"Error processing {link_url}: {str(e)}")
                else:
                    print("No additional URLs found in second pass analysis")
            
            except Exception as e:
                print(f"Second pass analysis failed: {str(e)}")

            # --- Chunking and saving ---
            if all_content:
                try:
                    combined_content = "\n".join(all_content)
                    # Optionally, re-run the generated content through the markdown fitter for consistency:
                    lines = combined_content.splitlines()
                    total_lines = len(lines)
                    
                    if total_lines > 3000:
                        chunk_size = 3000
                        num_chunks = (total_lines + chunk_size - 1) // chunk_size
                        print(f"\nContent is {total_lines} lines long. Splitting into {num_chunks} chunks...")
                        
                        base_name = output_file.rsplit('.', 1)[0]
                        for i in range(num_chunks):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, total_lines)
                            chunk_content = "\n".join(lines[start_idx:end_idx])
                            chunk_header = f"# Chunk {i+1} of {num_chunks}\n\n"
                            chunk_file = f"{base_name}_chunk_{i+1}.md"
                            with open(chunk_file, 'w', encoding='utf-8') as f:
                                f.write(chunk_header + chunk_content)
                            print(f"✓ Created chunk {i+1}: {chunk_file} ({end_idx - start_idx} lines)")
                        print(f"\n✓ Successfully split content from {len(self.processed_urls)} pages into {num_chunks} chunks")
                    else:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(combined_content)
                        print(f"\n✓ Successfully saved content from {len(self.processed_urls)} pages to {output_file} ({total_lines} lines)")
                except Exception as e:
                    print(f"\n✗ Error saving to file: {str(e)}")
            else:
                print("\n✗ No content was extracted")

if __name__ == "__main__":
    # Configure logging before anything else
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        handlers=[logging.FileHandler('logs/main.log'), logging.StreamHandler()]
    )
    
    load_dotenv()
    url = input("Enter URL: ").strip()
    main_prompt = os.getenv("MAIN_RESEARCH_PROMPT")
    if main_prompt:
        prompt = main_prompt
        print("Using MAIN_RESEARCH_PROMPT from .env")
    else:
        prompt = input("Enter research prompt (or leave empty to use default from .env): ").strip() or None
    asyncio.run(WebCrawler(verbose=True).crawl(url, prompt))
