import asyncio
import json
import re
from urllib.parse import urljoin, urlparse, urldefrag
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from pydantic import BaseModel, Field
from typing import List, Optional, Set, Dict
import os
import litellm

class ExtractedContent(BaseModel):
    relevant_urls: List[str] = Field(
        default_factory=list,
        description="List of URLs relevant to the research prompt"
    )
    relevance_reasons: List[str] = Field(
        default_factory=list,
        description="Brief reason for each URL's relevance"
    )
    main_topic: str = Field(
        default="Unknown topic - LLM failed to identify main theme",
        description="Primary topic identified from landing page"
    )

class WebCrawler:
    def __init__(self, verbose: bool = True):
        self.total_cost = 0.0
        # Verify OpenRouter API key format
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or not api_key.startswith("sk-or-"):
            raise ValueError(
                "Invalid OpenRouter API key format\n"
                "1. Get your key from https://openrouter.ai/keys\n"
                "2. Update .env file with your actual key\n"
                "3. Valid keys start with 'sk-or-'"
            )

        # Track LLM costs
        self.total_cost = 0.0
        
        # Initialize LLM extraction strategy with OpenRouter (only for landing page)
        self.llm_strategy = LLMExtractionStrategy(
            provider=f"{os.getenv('OPENROUTER_MODEL', 'openrouter/deepseek/deepseek-r1')}",
            api_token=api_key,
            extraction_type="schema",
            schema=ExtractedContent.model_json_schema(),
            instruction=(
                "Analyze the landing page and identify relevant URLs for documentation research. Be discriminating, only follow URL's that are directly related to the main topic and will be useful. We need to ensure our research is focused and relevant."
                "Return JSON with: 1) relevant_urls 2) brief relevance_reasons 3) main_topic. "
                "Only include URLs within the same documentation section/subfolder."
            ),
            api_base="https://openrouter.ai/api/v1",
            extra_args={
                "litellm_callbacks": [self._track_llm_cost],
                "headers": {
                    "HTTP-Referer": "https://github.com/your-project",
                    "X-Title": "Smart Documentation Extraction",
                    "Content-Type": "application/json",
                    "X-API-Key": os.getenv("OPENROUTER_API_KEY")
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

    def _track_llm_cost(self, kwargs, completion_response, start_time, end_time):
        """Callback to track LLM API costs"""
        try:
            cost = litellm.completion_cost(completion_response=completion_response)
            self.total_cost += cost
            print(f"\nLLM API Call Cost: ${cost:.6f} | Total: ${self.total_cost:.6f}")
        except Exception as e:
            print(f"Error tracking LLM cost: {str(e)}")

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

        # Configure the crawler with LLM strategy only for initial page
        self.llm_strategy.instruction = user_prompt
        llm_run_config = CrawlerRunConfig(
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

        # Non-LLM config for subsequent pages
        schema = {
            "name": "Documentation Research",
            "baseSelector": "body",
            "fields": [
                {
                    "name": "content",
                    "selector": "body",
                    "type": "markdown",
                    "transform": ["trim"]
                }
            ]
        }
        self.non_llm_strategy = JsonCssExtractionStrategy(schema, verbose=True)
        non_llm_config = CrawlerRunConfig(
            extraction_strategy=self.non_llm_strategy,
            cache_mode=CacheMode.BYPASS,
            markdown_generator=self.md_generator,
            verbose=False,
            process_iframes=True,
            word_count_threshold=1,
            excluded_tags=[],
            remove_overlay_elements=True,
            wait_for=self._get_wait_selector(url),
            page_timeout=15000
        )

        # Create link filter
        link_filter = self._create_link_filter(base_url)

        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                # First process landing page with LLM
                try:
                    initial_result = await crawler.arun(url=url, config=llm_run_config)
                except Exception as e:
                    print(f"\nRetrying landing page without wait condition...")
                    run_config = llm_run_config.clone()
                    run_config.wait_for = None
                    initial_result = await crawler.arun(url=url, config=run_config)

                if not initial_result.success:
                    print(f"\n✗ Failed to process landing page: {url}")
                    if hasattr(initial_result, 'error_message'):
                        print(f"Error: {initial_result.error_message}")
                    return


                # Process only LLM-recommended URLs from initial analysis
                validated_items = []
                content_data = initial_result.extracted_content
                error_message = None
                
                # Robust content data handling for different response formats
                try:
                    # Check if we have a method/function instead of raw data
                    if callable(content_data):
                        content_data = content_data()
                except Exception as e:
                    print(f"Error converting response data: {str(e)}")
                    content_data = f"Error decoding response: {str(e)}"
                
                # Force serialization for non-basic types
                if not isinstance(content_data, (str, dict, list, int, float, bool)):
                    content_data = str(content_data)

                # Handle error responses from LLM
                error_message = None
                
                # Handle list-type errors with better logging
                error_message = None
                if isinstance(content_data, list):
                    error_messages = []
                    for item in content_data:
                        if isinstance(item, dict):
                            message = item.get('content') or \
                                     item.get('error', {}).get('message') or \
                                     item.get('error', 'Unknown list error')
                            error_messages.append(f"List item error: {message}")
                    if error_messages:
                        error_message = "\n".join(error_messages[:3])  # Show first 3 errors only
                
                # Handle dict-type errors with better message extraction        
                elif isinstance(content_data, dict):
                    error_message = content_data.get('content') or \
                        content_data.get('error', {}).get('message') or \
                        content_data.get('error', 'Unknown dict error')
                # Handle raw response objects
                elif hasattr(content_data, 'text'):
                    try:
                        error_data = json.loads(content_data.text)
                        if isinstance(error_data, list):
                            error_message = error_data[0].get('content', 'LLM API Error')
                        else:
                            error_message = error_data.get('content', 
                                error_data.get('error', {}).get('message', 'LLM API Error'))
                    except:
                        error_message = str(content_data.text)
                # Handle string errors        
                elif isinstance(content_data, str):
                    error_message = content_data
                
                # Clean up error message formatting
                if error_message:
                    error_message = error_message.strip('[]"')  # Remove JSON array brackets and quotes

                # If we have an error message, create fallback content
                if error_message:
                    print(f"LLM API Error: {error_message}")
                    validated_items = [ExtractedContent(
                        relevant_urls=[url],
                        relevance_reasons=["Fallback to page content due to LLM error"],
                        main_topic=error_message
                    )]
                else:
                    # Try to parse valid response
                    try:
                        if isinstance(content_data, str):
                            content_data = json.loads(content_data)
                        elif hasattr(content_data, 'text'):  # Handle raw response objects
                            content_data = json.loads(content_data.text)
                        
                        if isinstance(content_data, list) and len(content_data) > 0:
                            content_data = content_data[0]

                        validated_items = [ExtractedContent.model_validate(content_data)]
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        print(f"Content parsing failed: {str(e)}")
                        validated_items = [ExtractedContent(
                            relevant_urls=[url],
                            relevance_reasons=["Fallback to page content due to parsing error"],
                            main_topic=f"Content extraction error: {str(e)}"
                        )]

                if not validated_items:
                    raise ValueError("Critical failure - no content could be extracted")

                # Process landing page content after validation
                if hasattr(initial_result, "html"):
                    # Use full markdown content from landing page
                    content = initial_result.markdown_v2.raw_markdown
                    if not content:
                        content = initial_result.html  # Fallback to raw HTML if markdown empty
                    if content:
                        print(f"\n✓ Successfully processed landing page: {url}")
                        # Get main_topic from validated content
                        main_topic = getattr(validated_items[0], 'main_topic', '')
                        header = f"# Landing Page Analysis\n**URL:** {url}\n**Main Topic:** {main_topic}\n\n"
                        all_content.append(header + content + "\n")
                        processed_urls.add(url)
                content_item = validated_items[0]
                print(f"LLM identified {len(content_item.relevant_urls)} relevant URLs")
                urls_to_process = set()
                
                # Add recommended URLs that pass our filters
                for url, reason in zip(content_item.relevant_urls, content_item.relevance_reasons):
                    # Clean up LLM-generated URLs and validate
                    clean_url = urljoin(base_url, url)  # Handle relative paths
                    clean_url, _ = urldefrag(clean_url)  # Remove fragments
                    clean_url = clean_url.replace('<', '').replace('>', '')  # Remove angle brackets
                    
                    # Validate URL structure
                    parsed = urlparse(clean_url)
                    if not parsed.scheme or not parsed.netloc:
                        print(f"  Skipping invalid URL: {clean_url}")
                        continue
                        
                    if clean_url not in processed_urls and link_filter(clean_url):
                        urls_to_process.add(clean_url)
                        
                print(f"\nProcessing {len(urls_to_process)} LLM-recommended pages")
                for link_url in urls_to_process:
                    try:
                        print(f"\nProcessing: {link_url}")
                        # Use non-LLM config for subsequent pages
                        result = await crawler.arun(url=link_url, config=non_llm_config)
                        
                        if result.success and hasattr(result, "html"):
                            # Extract content using LLM with relevance scoring
                            try:
                                extracted = json.loads(result.extracted_content)
                                
                                # Handle both single items and lists
                                if not isinstance(extracted, list):
                                    extracted = [extracted]

                                # Validate each item against our schema
                                valid_items = []
                                for item in extracted:
                                    try:
                                        validated = ExtractedContent.model_validate(item)
                                        valid_items.append(validated)
                                    except Exception as e:
                                        print(f"Invalid item skipped: {str(e)}")
                                        print(f"Invalid item content: {json.dumps(item, indent=2)[:300]}...")
                                
                                if not valid_items:
                                    raise ValueError("No valid extracted content found in LLM response")

                                # Aggregate all relevant URLs from valid items
                                all_relevant_urls = []
                                all_relevance_reasons = []
                                main_topics = set()
                                for item in valid_items:
                                    all_relevant_urls.extend(item.relevant_urls)
                                    all_relevance_reasons.extend(item.relevance_reasons)
                                    main_topics.add(item.main_topic)
                                
                                # Create consolidated content item
                                content_item = ExtractedContent(
                                    relevant_urls=all_relevant_urls,
                                    relevance_reasons=all_relevance_reasons,
                                    main_topic=", ".join(main_topics)
                                )
                            except (json.JSONDecodeError, ValueError, KeyError) as e:
                                print(f"Failed to parse LLM response: {str(e)}")
                                print(f"Aborting crawl due to extraction error: {str(e)}")
                                raise RuntimeError(f"Content extraction failed: {str(e)}") from e
                            # Process LLM response with URL recommendations
                            content_item = ExtractedContent(**extracted[0])
                            print(f"LLM identified {len(content_item.relevant_urls)} relevant URLs")
                                
                            # Crawl only LLM-recommended URLs
                            for url, reason in zip(content_item.relevant_urls, content_item.relevance_reasons):
                                if url not in processed_urls and link_filter(url):
                                    print(f"Crawling LLM-recommended URL: {url} ({reason})")
                                    result = await crawler.arun(url=url, config=run_config)
                                    if result.success:
                                        content = f"## Page Content\n**URL:** {url}\n\n{result.markdown_v2.raw_markdown}"
                                        all_content.append(content)
                                        processed_urls.add(url)
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
