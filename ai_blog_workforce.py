# -*- coding: utf-8 -*-
"""Hacker News Scraper and Markdown Writer Workforce with Summarizer Agent

This script defines a workforce with three agents:
1. Scraper Agent: Scrapes Hacker News for top stories and filters AI/LLM/ML related stories.
2. Writer Agent: Writes the scraped summaries to a markdown file.
3. Summarizer Agent: Summarizes the content of each story's URL and compiles summaries into a markdown file.
"""

# ----------------------------------
# Section 1: Import Dependencies & Configuration
# ----------------------------------

import os
import logging
import requests
from bs4 import BeautifulSoup
import textwrap
import json
import time
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce
from camel.types import ModelType, ModelPlatformType
from camel.toolkits import FunctionTool
from camel.tasks import Task
from camel.models import ModelFactory

# Configure logging
# Logging is turned off by setting it to CRITICAL to reduce console output
logging.basicConfig(level=logging.CRITICAL)

# Set up API keys (ensure these are set in your environment or prompt the user)
# For security, it's recommended to use environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print("[Main] Error: OPENAI_API_KEY not set in environment variables.")
    exit(1)

# ----------------------------------
# Section 2: Define Scraper Agent
# ----------------------------------


def is_ai_related(title: str) -> bool:
    """
    Determines if a story title is related to AI/LLM/ML by using GPT-4O model from OpenAI to interpret the context of the title.
    
    Args:
        title (str): The title of the Hacker News story.
        
    Returns:
        bool: True if the title is related to AI/LLM/ML, False otherwise.
    """
    print("[AI Analysis] Analyzing title for AI relevance...")
    prompt = f"Is the following title related to AI, machine learning, or deep learning? Title: '{title}'"
    
    # Set up the request body for OpenAI API
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "system", "content": "Determine if the title is related to AI, machine learning, or deep learning."},
                     {"role": "user", "content": prompt}]
    }
    
    # Send a POST request to the OpenAI API
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json=body
        )
        response.raise_for_status()  # This will raise an exception for HTTP errors
        ai_response = response.json()['choices'][0]['message']['content'].lower()
        
        # Interpret the AI's response
        if "yes" in ai_response or "related" in ai_response:
            print(f"[AI Analysis] Title '{title}' is AI-related.")
            return True
        else:
            print(f"[AI Analysis] Title '{title}' is not AI-related.")
            return False

    except Exception as e:
        print(f"[AI Analysis] Error during API call: {e}")
        return False  # Assume not related if there's an error

def scrape_hackernews() -> str:
    """Scrapes the top stories from Hacker News, filters AI/LLM/ML related stories, and returns them in JSON format.
    
    Returns:
        str: JSON string containing filtered stories with title, URL, and points
    """
    print("[Scraper Agent] Starting scrape_hackernews function.")
    try:
        print("1. Scraping Hacker News...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get("https://news.ycombinator.com/", headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')

        print("2. Parsing HTML content...")

        stories = []
        items = soup.select('.athing')
        print(f"3. Found {len(items)} story items.")

        for idx, item in enumerate(items[:20], start=1):  # Get top 20 stories
            print(f"4.{idx}. Processing story {idx}...")
            title_tag = item.select_one('.titleline > a')
            if not title_tag:
                print(f"   4.{idx}.1. Title tag not found.")
                problematic_html = item.prettify()
                print(f"   4.{idx}.2. Problematic HTML (first 500 chars):\n{problematic_html[:500]}...")
                title = "No title"
                url = "No URL"
            else:
                title = title_tag.get_text(strip=True)
                url = title_tag['href'] if 'href' in title_tag.attrs else "No URL"
                if not url.startswith("http"):
                    url = urljoin("https://news.ycombinator.com/", url)
                print(f"   4.{idx}.1. Title: {title}")
                print(f"   4.{idx}.2. URL: {url}")

            # Check if the title contains AI/LLM/ML keywords
            if not is_ai_related(title):
                print(f"   4.{idx}.3. Story does not match AI/LLM/ML keywords. Skipping.")
                continue

            # Points are in the next sibling <tr>
            subtext = item.find_next_sibling('tr').select_one('.subtext')
            if subtext:
                score_tag = subtext.select_one('.score')
                if score_tag:
                    points_text = score_tag.get_text(strip=True)
                    try:
                        points = int(points_text.replace(' points', '').replace(' point', ''))
                        print(f"   4.{idx}.3. Points extracted: {points}")
                    except ValueError:
                        points = 0
                        print(f"   4.{idx}.3. Points parsing failed. Defaulting to 0.")
                else:
                    points = 0
                    print(f"   4.{idx}.3. Score tag not found. Defaulting points to 0.")
            else:
                points = 0
                print(f"   4.{idx}.4. Subtext not found. Defaulting points to 0.")

            print(f"   4.{idx}.5. Points: {points}")
            stories.append({
                "title": title,
                "url": url,
                "points": points
            })

        print("5. Finished scraping stories.")
        print("[Scraper Agent] Completed scrape_hackernews function.")
        return json.dumps({"stories": stories})

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        print(f"[Scraper Agent] Error: HTTP Request failed: {e}")
        return json.dumps({"error": f"HTTP Request failed: {e}"})
    except Exception as e:
        logging.error(f"An error occurred while scraping: {e}")
        print(f"[Scraper Agent] Error: An error occurred: {e}")
        return json.dumps({"error": f"An error occurred: {e}"})

# Wrap the scraping function with FunctionTool
scraper_tool = FunctionTool(scrape_hackernews)

# Define the Scraper Agent
scraper_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Scraper Agent",
        content=textwrap.dedent("""
            You are a Scraper Agent. Your task is to:
            1. Scrape the top stories from Hacker News
            2. Filter for AI/LLM/ML related stories
            3. Extract the title, URL, and points for each AI-related story
            4. Return the filtered data in JSON format
        
            Example execution:
            ```python
            import json
            
            stories_json = scrape_hackernews()
            stories_data = json.loads(stories_json)
            
            if "error" in stories_data:
                print(f"Error: {stories_data['error']}")
                return
            
            # Process the AI-related stories
            for story in stories_data["stories"]:
                print(f"Title: {story['title']}")
                print(f"URL: {story['url']}")
                print(f"Points: {story['points']}")
                print("-" * 40)
            ```
        
            Always validate input/output and handle errors gracefully.
            Only return stories that are related to AI, LLM, or ML topics.
        """)
    ),
    model=ModelFactory.create(
        model_type=ModelType.GPT_4O_MINI,  # Corrected model type
        model_platform=ModelPlatformType.OPENAI,
    ),
    tools=[scraper_tool],
)

# ----------------------------------
# Section 3: Define Writer Agent
# ----------------------------------

import datetime

def write_markdown(stories_json: str) -> str:
    """Writes the scraped stories to a markdown file.
    
    Args:
        stories_json (str): JSON string containing stories to write
        
    Returns:
        str: Success or error message
    """
    print("[Writer Agent] Starting write_markdown function.")
    try:
        print("6. Writing stories to markdown file...")
        data = json.loads(stories_json)
        if "error" in data:
            print(f"[Writer Agent] Error: {data['error']}")
            return f"Error: {data['error']}"

        stories = data.get("stories", [])
        if not stories:
            print("[Writer Agent] No stories to write.")
            return "No stories to write."

        markdown_content = "# Hacker News Top AI/LLM/ML Stories\n\n"
        for idx, story in enumerate(stories, start=1):
            markdown_content += f"## {idx}. {story['title']}\n"
            markdown_content += f"- **URL:** [{story['url']}]({story['url']})\n"
            markdown_content += f"- **Points:** {story['points']} points\n\n"
            print(f"   6.{idx}. Written story {idx} to markdown.")

        # Generate timestamp for the output file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"blog-posts/{timestamp}_hackernews_summary.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logging.info(f"Markdown file '{output_file}' has been created successfully.")
        print("7. Markdown file has been successfully created.")
        print("[Writer Agent] Completed write_markdown function.")
        return f"Markdown file '{output_file}' created successfully."

    except json.JSONDecodeError:
        logging.error("Invalid JSON format.")
        print("[Writer Agent] Error: Invalid JSON format.")
        return "Error: Invalid JSON format."
    except Exception as e:
        logging.error(f"An error occurred while writing to markdown: {e}")
        print(f"[Writer Agent] Error: An error occurred: {e}")
        return f"Error: {e}"

# Wrap the writing function with FunctionTool
writer_tool = FunctionTool(write_markdown)

# Define the Writer Agent
writer_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Writer Agent",
        content=textwrap.dedent("""
            You are a Writer Agent. Your task is to:
            1. Take the JSON data of scraped Hacker News stories.
            2. Write a summary of the top stories in markdown format.
            3. Save the markdown content to a file.
        
            Example execution:
            ```python
            import json
            
            stories_json = scrape_hackernews()
            result = write_markdown(stories_json, "hackernews_summary.md")
            print(result)
            ```
        
            Always validate input/output and handle errors gracefully.
        """)
    ),
    model=ModelFactory.create(
        model_type=ModelType.GPT_4O_MINI,  # Corrected model type
        model_platform=ModelPlatformType.OPENAI,
    ),
    tools=[writer_tool],
)

# ----------------------------------
# Section 4: Define Summarizer Agent
# ----------------------------------

def summarize_sites(urls: list) -> str:
    """Takes a list of URLs, scrapes each site to extract main content, generates opinion pieces,
    and saves them as individual blog posts.
    
    Args:
        urls (list): List of URLs to analyze
        
    Returns:
        str: Success or error message
    """
    print("[Summarizer Agent] Starting summarize_sites function.")
    try:
        for url in urls:

            try:
                print(f"[Summarizer Agent] Processing {url}")
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract main content (basic implementation)
                content = ""
                for p in soup.find_all(['p', 'article', 'div', 'section']):
                    content += p.get_text() + "\n"
                
                if not content.strip():
                    print(f"[Summarizer Agent] No content found for {url}")
                    continue

                content = content[:100000]  # Limit content to 100,000 characters

                # Generate opinion piece using OpenAI
                prompt = f"""Analyze this technology article and write an opinion piece in Markdown format that includes:

                # TLDR
                A brief 2-3 sentence summary of the key points.

                # Analysis
                1. What the technology/webpage is about
                2. Pros and cons of the technology
                3. Why it's interesting
                4. Potential use cases and applications
                5. Technical analysis and insights
                6. Future implications
                7. Always reference the source

                # Follow-up Ideas
                List 3-5 potential follow-up topics or areas for further exploration.

                Article URL: {url}
                Article Content: {content}
                """

                # Create timestamp and sanitized title for filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                title = soup.title.string if soup.title else "Untitled"
                # Sanitize title for filename
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '-')[:50]  # Limit length
                
                filename = f"blog-posts/{timestamp}-{safe_title}.md"
                
                # Generate the opinion piece
                messages = [
                    {"role": "system", "content": "You are a technology analyst writing insightful opinion pieces about AI and technology."},
                    {"role": "user", "content": prompt}
                ]
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": messages,
                    }
                )
                
                if response.status_code != 200:
                    print(f"[Summarizer Agent] Error generating opinion for {url}: {response.text}")
                    continue
                
                opinion_piece = response.json()['choices'][0]['message']['content']
                
                # Create blog post with metadata
                blog_post = f"""---
title: {title}
url: {url}
date: {time.strftime("%Y-%m-%d %H:%M:%S")}
---

{opinion_piece}
"""
                
                # Save to file
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(blog_post)
                
                print(f"[Summarizer Agent] Saved opinion piece to {filename}")
                
            except Exception as e:
                print(f"[Summarizer Agent] Error processing {url}: {str(e)}")
                continue

        return "Successfully generated opinion pieces for valid URLs"
        
    except Exception as e:
        error_msg = f"Error in summarize_sites: {str(e)}"
        print(f"[Summarizer Agent] {error_msg}")
        return error_msg

# Wrap the summarization function with FunctionTool
summarizer_tool = FunctionTool(summarize_sites)

# Define the Summarizer Agent
summarizer_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Summarizer Agent",
        content=textwrap.dedent("""
            You are a Summarizer Agent. Your task is to:
            1. Receive a list of URLs from the Hacker News top stories but ONLY for AI/LLM/ML related stories.
            2. Scrape each URL to extract the main content.
            3. Generate a concise summary for each site.
            4. Compile all summaries into a consolidated markdown file.
        
            Example execution:
            ```python
            urls = [story['url'] for story in stories_data["stories"]]
            result = summarize_sites(urls, "site_summaries.md")
            print(result)
            ```
        
            Always validate input/output and handle errors gracefully.
        """)
    ),
    model=ModelFactory.create(
        model_type=ModelType.GPT_4O_MINI,  # Corrected model type
        model_platform=ModelPlatformType.OPENAI,
    ),
    tools=[summarizer_tool],
)

# ----------------------------------
# Section 5: Assemble the Workforce
# ----------------------------------

# Create the Workforce and add the Scraper, Writer, and Summarizer Agents
workforce = Workforce("Hacker News Summary Workforce")

workforce.add_single_agent_worker(
    "Scraper Agent - Scrapes top stories from Hacker News and filters AI/LLM/ML related stories",
    worker=scraper_agent
).add_single_agent_worker(
    "Summarizer Agent - Summarizes content from each story's URL, ONLY for AI/LLM/ML related stories",
    worker=summarizer_agent
).add_single_agent_worker(
    "Writer Agent - Writes filtered stories to markdown",
    worker=writer_agent
)

# ----------------------------------
# Section 6: Define the Task Flow
# ----------------------------------

def generate_hackernews_summary(workforce: Workforce) -> None:
    """Generates opinion pieces for AI-related Hacker News stories and saves them to the blog-posts directory."""
    print("[Task Flow] Starting generate_hackernews_summary function.")
    try:
        print("Starting Hacker News article analysis...")
        logging.info("Starting Hacker News article analysis...")

        # Step 1: Scrape AI-related stories
        print("8. Initiating scraping task...")
        scrape_task = Task(
            content="Scrape AI/LLM/ML related stories from Hacker News",
            id="scrape_hackernews"
        )
        scrape_result = workforce.process_task(scrape_task)
        print("8. Scraping task completed.")

        if not scrape_result:
            logging.error("Scraping task returned no result")
            print("[Task Flow] Error: Scraping task returned no result")
            return
            
        if not scrape_result.result:
            logging.error("Scraping task result is empty")
            print("[Task Flow] Error: Scraping task result is empty")
            return

        print(f"Debug - Scrape result: {scrape_result.result}")
            
        try:
            # Check for errors in scrape_result
            scrape_data = json.loads(scrape_result.result)
            if not isinstance(scrape_data, dict):
                logging.error(f"Invalid data format. Expected dict, got {type(scrape_data)}")
                print(f"[Task Flow] Error: Invalid data format. Expected dict, got {type(scrape_data)}")
                return
                
            if "error" in scrape_data:
                logging.error(f"Scraper Agent Error: {scrape_data['error']}")
                print(f"[Task Flow] Scraper Agent Error: {scrape_data['error']}")
                return
                
            if "stories" not in scrape_data:
                logging.error("No 'stories' key found in scraped data")
                print("[Task Flow] Error: No 'stories' key found in scraped data")
                return

            ai_stories = scrape_data["stories"]  # Stories are already filtered by the scraper
            if not ai_stories:
                print("No AI/LLM/ML related stories found unfortunately.")
                return
            print(f"9. Found {len(ai_stories)} AI/LLM/ML related stories.")

            # Step 2: Write to markdown
            print("10. Initiating writing task...")
            # Convert AI stories back to JSON
            ai_stories_json = json.dumps({"stories": ai_stories})
            write_task = Task(
                content="Write the AI/LLM/ML related stories to a markdown file",
                id="write_markdown",
                additional_info=ai_stories_json  # Passing the JSON data
            )
            write_result = workforce.process_task(write_task)
            print("10. Writing task completed.")

            if not write_result or not write_result.result:
                logging.error("Failed to write AI stories to markdown.")
                print("[Task Flow] Error: Failed to write AI stories to markdown.")
                return

            if write_result.result.startswith("Error"):
                logging.error(f"Writer Agent Error: {write_result.result}")
                print(f"[Task Flow] Writer Agent Error: {write_result.result}")
                return

            # Step 3: Generate opinion pieces for each story
            print("11. Initiating opinion piece generation...")
            urls = [story['url'] for story in ai_stories if story['url'] != "No URL"]
            if not urls:
                print("No valid URLs to analyze.")
                return
            summarize_task = Task(
                content="Generate opinion pieces for each AI-related story",
                id="summarize_sites",
                additional_info=json.dumps(urls)  # Passing the list of URLs as JSON
            )
            summarize_result = workforce.process_task(summarize_task)
            print("11. Opinion piece generation completed.")

            if not summarize_result or not summarize_result.result:
                logging.error("Failed to generate opinion pieces.")
                print("[Task Flow] Error: Failed to generate opinion pieces.")
                return

            if summarize_result.result.startswith("Error"):
                logging.error(f"Summarizer Agent Error: {summarize_result.result}")
                print(f"[Task Flow] Summarizer Agent Error: {summarize_result.result}")
                return

            logging.info("Opinion pieces have been successfully generated.")
            print("Opinion pieces have been successfully generated and saved to the blog-posts directory.")
            print("[Task Flow] Completed generate_hackernews_summary function.")

        except json.JSONDecodeError as e:
            logging.critical(f"Failed to parse JSON: {e}")
            print(f"[Task Flow] Critical Error: Failed to parse JSON: {e}")
        except Exception as e:
            logging.critical(f"An unexpected error occurred: {e}")
            print(f"[Task Flow] Critical Error: An unexpected error occurred: {e}")

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        print(f"[Task Flow] Critical Error: An unexpected error occurred: {e}")

# ----------------------------------
# Section 7: Execute the Task
# ----------------------------------

if __name__ == "__main__":
    print("[Main] Starting script execution.")
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create blog-posts directory if it doesn't exist
    os.makedirs("blog-posts", exist_ok=True)
    
    # Initialize the workforce
    print("[Main] Initializing workforce...")
    workforce = Workforce("Hacker News Opinion Piece Generator")
    print("[Main] Adding agents to workforce...")
    
    # Add agents with debug output
    print("[Main] Adding Scraper Agent...")
    workforce.add_single_agent_worker(
        "Scraper Agent - Scrapes top stories from Hacker News and filters AI/LLM/ML related stories",
        worker=scraper_agent
    )
    print("[Main] Adding Writer Agent...")
    workforce.add_single_agent_worker(
        "Writer Agent - Writes filtered stories to markdown",
        worker=writer_agent
    )
    print("[Main] Adding Summarizer Agent...")
    workforce.add_single_agent_worker(
        "Summarizer Agent - Generates opinion pieces for AI/ML/LLM stories",
        worker=summarizer_agent
    )
    
    print("[Main] Starting opinion piece generation...")
    generate_hackernews_summary(workforce)
    print("[Main] Script execution finished.")
