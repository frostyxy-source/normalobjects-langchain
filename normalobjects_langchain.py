import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.tools import tool
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict
from langchain_classic.callbacks.base import BaseCallbackHandler

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# STEP 2: TOOLS
# ============================================================

@tool
def consult_demogorgon(complaint: str) -> str:
    """Get the Demogorgon's perspective on a complaint about the Upside Down."""
    import random
    responses = [
        f"The Demogorgon tilts its head. It seems confused by '{complaint}'. Perhaps the issue is that you're thinking in three dimensions?",
        f"The Demogorgon makes a sound that might be agreement. It suggests that the problem might be temporal - things work differently in the Upside Down's time.",
        f"The Demogorgon appears to be eating something. It doesn't seem to understand the concept of '{complaint}' - maybe consistency isn't a priority there?"
    ]
    return random.choice(responses)

@tool
def check_hawkins_records(query: str) -> str:
    """Search Hawkins historical records for information."""
    records = {
        "portal": "Records show portals have opened on various dates with no clear pattern.",
        "monsters": "Historical records indicate creatures from the Upside Down behave differently based on environmental factors.",
        "psychics": "Records show that psychic abilities vary greatly.",
        "electricity": "Hawkins has a history of electrical anomalies."
    }
    for key, value in records.items():
        if key in query.lower():
            return value
    return f"Records don't contain specific information about '{query}', but many unexplained events have occurred in Hawkins."

@tool
def cast_interdimensional_spell(problem: str, creativity_level: str = "medium") -> str:
    """Suggest a creative interdimensional spell to fix a problem."""
    import random
    creativity_multiplier = {"low": 1, "medium": 2, "high": 3}[creativity_level]
    spells = [
        f"Try chanting 'Becma Becma Becma' three times while holding a Walkman. This might recalibrate the interdimensional frequencies related to: {problem}",
        f"Create a salt circle and place a compass in the center. The magnetic anomalies might help stabilize: {problem}",
        f"Play 'Running Up That Hill' backwards at the exact location of the issue. The temporal resonance could fix: {problem}",
        f"Gather three items: a lighter, a compass, and something personal. Arrange them in a triangle while thinking about: {problem}.",
    ]
    selected = random.sample(spells, min(creativity_multiplier, len(spells)))
    return "\n".join(selected)

@tool
def gather_party_wisdom(question: str) -> str:
    """Ask the D&D party (Mike, Dustin, Lucas, Will) for their collective wisdom."""
    party_responses = {
        "portal": "Mike: 'Portals are unpredictable!' Dustin: 'They follow the Mind Flayer's activity.'",
        "monsters": "Lucas: 'Demogorgons are territorial.' Will: 'They can sense fear and strong emotions.'",
        "psychics": "Mike: 'El's powers are connected to her emotional state.' Dustin: 'Limited by her energy.'",
        "electricity": "Lucas: 'The Upside Down interferes with electrical systems.' Dustin: 'It's like a feedback loop.'"
    }
    for key, response in party_responses.items():
        if key in question.lower():
            return response
    return "The party huddles together. Mike: 'This is a tough one.' Dustin: 'We need more information.'"

tools = [consult_demogorgon, check_hawkins_records, cast_interdimensional_spell, gather_party_wisdom]

print(f"Created {len(tools)} creative tools:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description[:60]}...")

# ============================================================
# STEP 3: AGENT
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Becma, the creative complaint handler for the Downside-Up Complaint Bureau!
    Always use your tools to investigate complaints before giving a final answer.
    Be creative, entertaining, and thorough in your investigations!"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

print("Agent created successfully!")

# ============================================================
# STEP 5: TRACKER (defined before anything runs!)
# ============================================================

class ToolUsageTracker(BaseCallbackHandler):
    """Track tool usage for analysis"""
    def __init__(self):
        self.usage_count = {tool.name: 0 for tool in tools}
        self.tool_sequences = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        if tool_name in self.usage_count:
            self.usage_count[tool_name] += 1
            self.tool_sequences.append(tool_name)

    def get_statistics(self):
        return {
            "total_tool_calls": sum(self.usage_count.values()),
            "tool_counts": self.usage_count,
            "most_used": max(self.usage_count.items(), key=lambda x: x[1])[0] if self.usage_count else None,
            "tool_sequences": self.tool_sequences
        }

# Create tracker BEFORE any complaints run
tracker = ToolUsageTracker()
agent_executor.callbacks = [tracker]

# ============================================================
# STEP 4: COMPLAINTS
# ============================================================

complaints = [
    "Why do demogorgons sometimes eat people and sometimes don't?",
    "The portal opens on different days—is there a schedule?",
    "Why can some psychics see the Downside Up and others can't?",
    "How come a bunch of kids outsmart the US Army?",
]

def handle_complaint(complaint: str) -> str:
    print(f"\n{'='*60}")
    print(f"COMPLAINT: {complaint}")
    print(f"{'='*60}\n")
    result = agent_executor.invoke(
        {"input": complaint},
        config={"callbacks": [tracker]}
    )
    return result["output"]

print("Testing agent with sample complaints...\n")
for complaint in complaints[:4]:
    response = handle_complaint(complaint)
    print(f"\nRESPONSE: {response}\n")

# ============================================================
# ANALYSIS
# ============================================================

print("\n=== Tool Usage Analysis ===")
stats = tracker.get_statistics()
print(f"Total tool calls: {stats['total_tool_calls']}")
print(f"Tool usage counts: {stats['tool_counts']}")
print(f"Most used tool: {stats['most_used']}")
print(f"\nTool sequence:")
print(" -> ".join(stats['tool_sequences']))