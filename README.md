
## Setup 

### Prerequisites

- Ensure you're using Python 3.11 - 3.13.
- [uv](https://docs.astral.sh/uv/) package manager or [pip](https://pypi.org/project/pip/)
- OpenAI API key
- Node.js and npx (required for MCP server in notebook 3):
```bash
# Install Node.js (includes npx)
# On macOS with Homebrew:
brew install node

# On Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation:
node --version
npx --version
```

### Installation

Download the course repository

```bash
# Clone the repo, cd to 'python' directory
git clone https://github.com/msharan/data_science_ai_agent.git
```

Make a copy of example.env

```bash
# Create .env file
cp example.env .env
```

Insert API keys directly into .env file, OpenAI (required) and [LangSmith](#getting-started-with-langsmith) (optional)

```bash
# Add OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
# The course is written with OpenAI models, but you can choose others if you prefer. 
# Be sure to add the key and modify the code to call your preferred model
#ANTHROPIC_API_KEY=your_anthropic_api_key_here_if_you_prefer

# Optional API key for LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langgraph-py-essentials
# If you are on the EU instance:
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com

```

Make a virtual environment and install dependancies
```bash
# Create virtual environment and install dependancies
uv sync
```

Run notebooks

```bash
# Run Jupyter notebooks directly with uv
uv run jupyter lab

# Or activate the virtual environment if preferred
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
jupyter lab
```

Optional: Setup [LangSmith Studio](https://docs.langchain.com/oss/python/langchain/studio)

```bash
# copy the .env file you created above to the studio directory
cp .env ./studio/.

#to run
langgraph dev
```
