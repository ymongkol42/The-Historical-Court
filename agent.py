import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

# Import ADK libraries
from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.models import Gemini
from google.adk.tools import exit_loop
from google.genai import types

# Import LangChain & Wikipedia libraries
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# --- 1. SETUP & CONFIGURATION ---

# Load .env
if os.path.exists("parent_and_subagents/.env"):
    load_dotenv("parent_and_subagents/.env")
elif os.path.exists(".env"):
    load_dotenv(".env")

# Config variables
use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
model_name = os.getenv("MODEL", "gemini-1.5-flash")
api_key = os.getenv("GOOGLE_API_KEY")

RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

def get_gemini_model():
    if use_vertex and use_vertex.upper() == "TRUE" and project_id and location:
        print(f"✅ Using Vertex AI (Project: {project_id}, Loc: {location})")
        return Gemini(model=model_name, vertexai=True, project=project_id, location=location, retry_options=RETRY_OPTIONS)
    elif api_key:
        print("✅ Using Standard API Key")
        return Gemini(model=model_name, api_key=api_key, retry_options=RETRY_OPTIONS)
    else:
        raise ValueError("❌ Missing Configuration! Check .env")

# Setup Logging
try:
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
except Exception:
    pass

# --- 2. TOOLS DEFINITION ---

def append_to_state(tool_context: ToolContext, field: str, response: str) -> dict[str, str]:
    """Appends data to a specific state key (e.g., pos_data, neg_data)."""
    existing_state = tool_context.state.get(field, [])
    if isinstance(existing_state, str):
        existing_state = [existing_state]
    
    # Add new data
    new_data = existing_state + [response]
    tool_context.state[field] = new_data
    
    logging.info(f"[Added to {field}] Length: {len(str(response))}")
    return {"status": "success", "message": f"Data appended to {field}"}

def write_verdict_file(tool_context: ToolContext, filename: str, content: str) -> dict[str, str]:
    """Saves the final verdict to a text file."""
    directory = "court_records"
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding='utf-8') as f:
        f.write(content)
    return {"status": "success", "path": target_path}

# --- 3. AGENT DEFINITIONS ---

# --- Step 2: The Investigation (Parallel Agents) ---

# Agent A: The Admirer (Positive Side)
admirer = Agent(
    name="admirer",
    model=get_gemini_model(),
    description="Finds positive achievements and legacies.",
    instruction="""
    ROLE: You are 'The Admirer'. You verify the greatness of the subject.
    
    INPUT CONTEXT:
    - Subject: { TOPIC? }
    - Judge's Feedback (if any): { JUDGE_FEEDBACK? }

    INSTRUCTIONS:
    1. Search Wikipedia for the Subject. **ALWAYS append keywords** like "achievements", "success", "legacy", "honors" to your search query to find positive aspects.
    2. If there is JUDGE_FEEDBACK saying data is missing, search specifically for what is asked.
    3. Summarize the **Positive** facts found.
    4. Use tool 'append_to_state' to save your summary to the key 'pos_data'.
    
    IMPORTANT: Do NOT report controversies. Focus ONLY on the good side.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state
    ]
)

# Agent B: The Critic (Negative Side)
critic = Agent(
    name="critic",
    model=get_gemini_model(),
    description="Finds controversies, failures, and criticisms.",
    instruction="""
    ROLE: You are 'The Critic'. You expose the flaws of the subject.
    
    INPUT CONTEXT:
    - Subject: { TOPIC? }
    - Judge's Feedback (if any): { JUDGE_FEEDBACK? }

    INSTRUCTIONS:
    1. Search Wikipedia for the Subject. **ALWAYS append keywords** like "controversy", "criticism", "failures", "war crimes", "scandals" to your search query.
    2. If there is JUDGE_FEEDBACK saying data is missing, search specifically for what is asked.
    3. Summarize the **Negative** facts found.
    4. Use tool 'append_to_state' to save your summary to the key 'neg_data'.
    
    IMPORTANT: Do NOT report achievements. Focus ONLY on the bad side.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state
    ]
)

investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[admirer, critic],
    description="Investigates both sides of the history simultaneously."
)

# --- Step 3: The Trial (The Judge) ---

judge = Agent(
    name="judge",
    model=get_gemini_model(),
    description="Evaluates evidence and controls the loop.",
    instruction="""
    ROLE: You are 'The Judge'. You ensure a fair trial by checking if we have enough information from both sides.

    EVIDENCE:
    - Positive Data: { pos_data? }
    - Negative Data: { neg_data? }
    
    INSTRUCTIONS:
    1. Analyze the 'pos_data' and 'neg_data'.
    2. Check for **Balance**: Is one side detailed while the other is empty or too short?
    3. Check for **Sufficiency**: Is there enough substance to form a verdict?
    
    DECISION LOGIC:
    - IF data is missing or unbalanced:
      - Use 'append_to_state' to write specific instructions to 'JUDGE_FEEDBACK' (e.g., "Critic, find more about his later years").
      - Do NOT exit the loop. The system will loop back to the investigation team.
      
    - IF data is balanced and sufficient:
      - Use 'exit_loop' tool to end the trial.
      - Pass a brief confirmation message (e.g., "Evidence accepted").
    """,
    tools=[append_to_state, exit_loop]
)

# Loop Controller
historical_court_loop = LoopAgent(
    name="historical_court_loop",
    description="Iterates the investigation until the Judge is satisfied.",
    sub_agents=[
        investigation_team, # Step 2
        judge               # Step 3
    ],
    max_iterations=3 # Limit loops to prevent infinite running
)

# --- Step 4: The Verdict (Output) ---

verdict_writer = Agent(
    name="verdict_writer",
    model=get_gemini_model(),
    description="Writes the final neutral report.",
    instruction="""
    ROLE: You are the Court Scribe.
    
    INPUT:
    - Subject: { TOPIC? }
    - Positive Evidence: { pos_data? }
    - Negative Evidence: { neg_data? }
    
    INSTRUCTIONS:
    1. Analyze all evidence provided.
    2. Write a **Neutral Verdict Report** in THAI language.
    3. The report must contain:
       - Introduction
       - The Admirer's Argument (Pros)
       - The Critic's Argument (Cons)
       - Final Neutral Conclusion
    4. Use 'write_verdict_file' tool to save it.
       - Filename: "{TOPIC}_verdict.txt" (replace spaces with underscores).
       - Content: The full report.
    """,
    tools=[write_verdict_file]
)

# --- Main Orchestrator ---

court_system = SequentialAgent(
    name="court_system",
    description="Manages the flow of the historical court.",
    sub_agents=[
        historical_court_loop, # Run the trial loop
        verdict_writer         # Write final report
    ]
)

root_agent = Agent(
    name="clerk",
    model=get_gemini_model(),
    description="Step 1: The Inquiry",
    instruction="""
    1. Greet the user to 'The Historical Court'.
    2. Ask the user for a **Historical Figure or Event** to put on trial.
    3. Use 'append_to_state' to save the user's input to the key 'TOPIC'.
    4. Transfer control to the 'court_system'.
    """,
    tools=[append_to_state],
    sub_agents=[court_system]
)

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print(f"⚖️ The Historical Court is in session! (Model: {model_name})")
    print("System ready. Waiting for user input...")
    
    session_state = {}
    
    try:
        # Start conversation
        response = root_agent.query(user_input=None, state=session_state)
        print(f"\nClerk: {response.output_text}")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "ออก"]:
                break
                
            # Send input to agents
            response = root_agent.query(user_input=user_input, state=session_state)
            print(f"\nSystem: {response.output_text}")
            
            # Check if file is created
            if "court_records" in str(response.output_text) or any("_verdict.txt" in str(v) for v in session_state.values()):
                print("\n--- Court Adjourned. Verdict saved. ---")
                # Optional: break or continue for new topic
                
    except Exception as e:
        print(f"\n❌ Error: {e}")