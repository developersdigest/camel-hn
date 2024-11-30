# README.md

# Hacker News Opinion Piece Generator

This project is a Python-based application that scrapes the top stories from Hacker News, filters for AI/LLM/ML related stories, and generates opinion pieces in markdown format. It utilizes a workforce of agents to perform scraping, writing, and summarizing tasks.

## Requirements

- Python 3.10
- Virtual Environment (venv)

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/developersdigest/camel-hn.git
   cd camel-hn
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3.10 -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables:**
   Create a `.env` file in the root directory and add your OpenAI API key and any other necessary environment variables:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

To run the application, execute the following command:
