from autogen import AssistantAgent, UserProxyAgent
import arxiv
import requests
from io import BytesIO
import PyPDF2
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get('OPENAI_API_KEY')

# LLM Configuration
def get_llm_config():
    # Ensure API key is available
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return {
        "api_key": API_KEY,
        "model": "gpt-4-turbo",
        "temperature": 0.7,
        "max_tokens": 1000
    }

# Create Assistant Agent
def create_assistant_agent():
    return AssistantAgent(
        name="research_assistant",
        system_message="""
        You are an AI assistant specialized in research tasks, including paper recommendations, summarization, trend analysis, idea generation, and evaluate idea feasibility.
        All responses must adhere to the following guidelines:

        1) Do not fabricate information in the absence of sufficient evidence, and clearly indicate "알 수 없습니다" or "잘 모르겠습니다" if uncertain.
        2) Verify possible information step-by-step before responding, marking ambiguous or unclear sources as "확실하지 않음."
        3) Base final responses only on verified information, keeping answers concise. If speculation is necessary, disclose it by stating "추측입니다."
        4) If the user's query is unclear or requires further information, first request additional context or details from the user.
        5) Do not confidently assert unverified facts and provide evidence if necessary, including sources or references when available.
        6) For every answer, specify supporting information with references or summarized related links and materials wherever possible.

        # Steps
        - **Paper Recommendations:**
          1. Identify the specific field or topic area the user is interested in.
          2. Search for recent and relevant papers within that field.
          3. Filter and rank them based on relevance, publication date, and impact factor.
          4. Provide a list of recommended papers with a brief description of each.

        - **Summarization:**
          1. Extract key points and findings from the provided paper or research material.
          2. Highlight significant contributions and conclusions.
          3. Write a concise summary that communicates the main insights.

        - **Trend Analysis:**
          1. Gather data or publications related to a specific research domain over time.
          2. Identify patterns, common themes, emerging topics, and shifts in focus.
          3. Present an analysis that explains these trends.

        - **Idea Generation:**
         1. Understand the user's area of interest and goals.
         2. Brainstorm and gather inspiration from recent publications, trends, and gaps in the literature.
         3. Present new research ideas or questions that can be pursued further.

        # Output Format

        - Responses should be in paragraph form, clearly structured and organized according to the task type.
        - For lists, use bullet points or numbered lists where appropriate.
        - Ensure any visualizations are described in detail with clear explanations for what they represent.
        - Use a formal and informative tone suitable for academic or professional contexts.

        # Notes

        - Ensure recommendations are current and papers are from credible sources.
        - For trend analysis, consider incorporating historical data and future predictions when possible.
        - When generating ideas, ensure they are feasible and backed by current research to a reasonable extent.""",
        llm_config=get_llm_config()
    )


# Create User Proxy Agent
def create_user_proxy():
    return UserProxyAgent(
        name="Admin",
        system_message="You are the user proxy handling requests and interacting with the AI assistant.",
        human_input_mode="ALWAYS",
        default_auto_reply="Reply 'TERMINATE' if the task is done.",
        code_execution_config={"use_docker": False} # Set to False if Docker isn't used or configured
    )

# Task functions
def recommend_papers_tool(query: str, year: int = None, limit: int = 5) -> str:
    """Recommends research papers from arXiv based on a query, optional year, and limit."""
    try:
        search_query = f"{query}"
        if year:
            # Ensure year is handled correctly in the query format if needed
            search_query += f" AND submittedDate:[{year}0101 TO {year}1231]"

        client = arxiv.Client()
        search = arxiv.Search(query=search_query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)

        results = list(client.results(search)) # Fetch results

        if not results:
            return f"No papers found for '{query}'" + (f" in {year}." if year else ".")

        papers = [
            f"Title: {result.title}\n"
            f"Authors: {', '.join([author.name for author in result.authors])}\n"
            f"Abstract: {result.summary}\n"
            f"Submitted Date: {result.updated.date().isoformat()}\n"
            f"Link: {result.entry_id}" # Add link for reference
            for result in results
        ]
        return f"Recommended papers for '{query}':\n\n" + "\n\n".join(papers)
    except Exception as e:
        return f"Error while recommending papers: {e}"

def summarize_pdf_tool(url: str) -> str:
    """Summarizes the text content of a PDF from a given URL."""
    try:
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("NLTK 'punkt' tokenizer not found. Downloading...")
            nltk.download('punkt')

        response = requests.get(url, timeout=30) # Added timeout
        response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)

        memory_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(memory_file)
        pdf_text = "".join(page.extract_text() or "" for page in pdf_reader.pages) # Handle None from extract_text

        if not pdf_text.strip():
            return "No text could be extracted from the PDF or the PDF is empty."

        parser = PlaintextParser.from_string(pdf_text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        total_sentences = len(parser.document.sentences)

        # Adjust summary length: 20% of sentences, but at least 3 and at most 10 (or adjust as needed)
        summary_length = max(3, min(10, total_sentences // 5))

        if total_sentences < 3:
            summary_length = total_sentences # Summarize all if less than 3 sentences

        summary = summarizer(parser.document, sentences_count=summary_length)
        return "PDF Summary:\n" + " ".join(str(sentence) for sentence in summary)

    except requests.exceptions.RequestException as e:
        return f"Error fetching PDF from URL: {e}"
    except PyPDF2.errors.PdfReadError as e:
        return f"Error reading PDF file: {e}. The file might be corrupted or password-protected."
    except Exception as e:
        return f"An unexpected error occurred during PDF summarization: {e}"

def extract_research_trends(query: str, year: int = 2025, limit: int = 10) -> str:
    """Analyzes research trends based on keywords extracted from recent paper abstracts."""
    try:
        # Use the API client initialized globally if needed, or create locally
        client = OpenAI(api_key=API_KEY)
        if not client.api_key:
            raise ValueError("OpenAI API key is not configured.")

        paper_recommendations = recommend_papers_tool(query, year, limit)

        # Check if recommendations contain an error message
        if paper_recommendations.startswith("Error") or paper_recommendations.startswith("No papers found"):
            return f"Could not fetch papers for trend analysis: {paper_recommendations}"

        # Extract abstracts more robustly
        abstracts = []
        papers_data = paper_recommendations.split("\n\n")[1:] # Skip the header
        for paper_block in papers_data:
            lines = paper_block.split('\n')
            for line in lines:
                if line.startswith("Abstract: "):
                    abstracts.append(line.replace("Abstract: ", "").strip())
                    break # Found abstract for this paper

        if not abstracts:
            return "No abstracts found in the recommended papers. Unable to analyze trends."

        # Handle case with fewer abstracts than max_features
        actual_max_features = min(10, len(abstracts))
        if actual_max_features == 0:
            return "No abstracts available to extract keywords from."

        vectorizer = TfidfVectorizer(stop_words='english', max_features=actual_max_features)
        vectorizer.fit_transform(abstracts) # Fit TF-IDF model
        top_keywords = vectorizer.get_feature_names_out()

        if not top_keywords.any():
            return "Could not extract meaningful keywords from the abstracts."

        prompt = f"""
        Based on the following keywords extracted from recent research papers related to '{query}': {', '.join(top_keywords)}.
        Please provide a concise analysis of the current research trends in this area.
        """

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return "Research Trend Analysis:\n" + response.choices[0].message.content

    except Exception as e:
        return f"Error extracting research trends: {e}"

def generate_research_idea(query: str, year: int = 2025, limit: int = 5) -> str: # Reduced limit for faster processing
    """Generates novel research ideas based on potential gaps in recent papers."""
    try:
        client = OpenAI(api_key=API_KEY)
        if not client.api_key:
            raise ValueError("OpenAI API key is not configured.")

        paper_recommendations = recommend_papers_tool(query, year, limit)

        if paper_recommendations.startswith("Error") or paper_recommendations.startswith("No papers found"):
            return f"Could not fetch papers for idea generation: {paper_recommendations}"

        # Extract abstracts robustly
        abstracts_content = []
        papers_data = paper_recommendations.split("\n\n")[1:]
        for paper_block in papers_data:
            lines = paper_block.split('\n')
            abstract_line = next((line for line in lines if line.startswith("Abstract: ")), None)
            if abstract_line:
                abstracts_content.append(abstract_line.replace("Abstract: ", "").strip())

        if not abstracts_content:
            return "No abstracts found to generate research ideas from."

        # Use only a few abstracts for the prompt to stay within token limits if necessary
        abstracts_for_prompt = "\n\n".join(abstracts_content[:3]) # Use up to 3 abstracts

        prompt = f"""
        Please review the following abstracts from recent papers on '{query}':
        --- START ABSTRACTS ---
        {abstracts_for_prompt}
        --- END ABSTRACTS ---

        Based *only* on the information and potential limitations hinted at in these abstracts:
        1) Identify potential research gaps or areas needing further exploration.
        2) Propose one or two novel research ideas addressing these gaps.
        3) Briefly explain the potential significance or contribution of each idea.
        """

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return "Generated Research Idea(s):\n" + response.choices[0].message.content

    except Exception as e:
        return f"Error generating research ideas: {e}"

def evaluate_feasibility(research_idea: str) -> str:
    """Evaluates the feasibility of a given research idea using predefined criteria."""
    try:
        client = OpenAI(api_key=API_KEY)
        if not client.api_key:
            raise ValueError("OpenAI API key is not configured.")

        prompt = f"""
        Evaluate the feasibility of the following research idea:
        --- RESEARCH IDEA ---
        {research_idea}
        --- END RESEARCH IDEA ---

        Assess the idea based on these criteria:
        1) **Technical Readiness Level (TRL):** Estimate on a scale of 1 (basic principles) to 9 (proven system). Provide a score and brief justification.
        2) **Data Availability:** Estimate the ease of obtaining necessary data (1: Very difficult/non-existent to 10: Readily available). Provide a score and reasoning.
        3) **Research Complexity:** Estimate the overall difficulty (1: Straightforward to 10: Highly complex/challenging). Provide a score and explain why.

        Provide a final summary assessment of the overall feasibility.
        """

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return "Feasibility Evaluation:\n" + response.choices[0].message.content

    except Exception as e:
        return f"Error evaluating feasibility: {e}"

# Main execution block
def chat_loop(user_proxy, assistant_agent):
    """Runs the main interactive chat loop."""
    print("--- Welcome to the Interactive Research Assistant ---")
    print("Ask me to recommend papers, summarize PDFs, analyze trends, generate ideas, or evaluate feasibility.")
    print("Examples:")
    print("  Recommend papers on 'quantum machine learning' from 2024.")
    print("  Summarize PDF from URL: <pdf_url>")
    print("  Analyze research trends for 'large language models'.")
    print("  Generate research ideas based on 'AI in drug discovery'.")
    print("  Evaluate feasibility: <paste the research idea here>")
    print("\nType 'exit' or 'quit' to end.")
    print("-" * 55)

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\nAssistant: Exiting the assistant. Goodbye!")
                break

            if not user_input:
                continue

            # Initiate chat and get response
            # Using chat_messages to maintain history might be better for context
            # For simplicity, initiating a new chat each time here.
            chat_result = user_proxy.initiate_chat(
                assistant_agent,
                message=user_input, # Pass the user input directly
                # clear_history=False # Consider managing history if context is needed across turns
            )

            # Print the last message from the chat, assuming it's the assistant's final reply
            if chat_result and chat_result.chat_history:
                print(f"\nAssistant:\n{chat_result.chat_history[-1]['content']}")
            else:
                # Handle cases where initiate_chat might not return expected result
                # This might happen if the flow terminates unexpectedly.
                print("\nAssistant: I couldn't process that request. Please try again.")


        except KeyboardInterrupt:
            print("\nAssistant: Received interrupt signal. Exiting gracefully...")
            break
        except Exception as e:
            print(f"\nAn error occurred in the chat loop: {e}")
            # Decide whether to continue or break on general errors
            # continue


if __name__ == "__main__":
    # Initialize agents
    try:
        assistant = create_assistant_agent()
        user_proxy_agent = create_user_proxy()

        # Register tools for a function calling
        # For AssistantAgent (LLM decides when to call)
        assistant.register_for_llm(name="recommend_papers", description="Recommend research papers from arXiv based on query, year, limit.")(recommend_papers_tool)
        assistant.register_for_llm(name="summarize_pdf", description="Summarize the text content of a PDF from a URL.")(summarize_pdf_tool)
        assistant.register_for_llm(name="extract_research_trends", description="Analyze research trends using keywords from recent paper abstracts.")(extract_research_trends)
        assistant.register_for_llm(name="generate_research_idea", description="Generate novel research ideas based on gaps in recent papers.")(generate_research_idea)
        assistant.register_for_llm(name="evaluate_feasibility", description="Evaluate the feasibility of a research idea based on TRL, data availability, and complexity.")(evaluate_feasibility)

        # For UserProxyAgent (Agent executes the call)
        user_proxy_agent.register_for_execution(name="recommend_papers")(recommend_papers_tool)
        user_proxy_agent.register_for_execution(name="summarize_pdf")(summarize_pdf_tool)
        user_proxy_agent.register_for_execution(name="extract_research_trends")(extract_research_trends)
        user_proxy_agent.register_for_execution(name="generate_research_idea")(generate_research_idea)
        user_proxy_agent.register_for_execution(name="evaluate_feasibility")(evaluate_feasibility)

        # Start the chat loop
        chat_loop(user_proxy_agent, assistant)

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as ex:
        print(f"Failed to initialize the application: {ex}")