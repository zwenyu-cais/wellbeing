from typing import Dict, List, Optional
import re
from ..utils import create_agent, generate_responses

SUMMARIZER_SYSTEM_PROMPT = """You are a precise and consistent summarizer that extracts key elements from text.
Your goal is to identify and extract:
1. Objects/nouns that are being interacted with
2. Actions/verbs being performed
3. Locations mentioned
4. State changes or transformations

Be extremely consistent in your extractions and maintain a minimal, clear format."""

SUMMARIZE_PROMPT = """Extract the key elements from the following text, focusing on objects, actions, and state changes.
Keep only the most important elements and maintain clarity.

Text: {text}

Format your response as:
Objects: [comma-separated list]
Actions: [comma-separated list]
Locations: [comma-separated list]
Changes: [key state changes]"""

class SummarizerAgent:
    def __init__(self, model_key: str = "gpt-35-turbo", api_key: Optional[str] = None):
        """Initialize the summarizer agent with a very low temperature for consistency."""
        config = {
            'max_tokens': 150,  # Summaries should be concise
            'temperature': 0.1,  # Very low temperature for consistency
            'api_key': api_key
        }
        self.agent = create_agent(model_key=model_key, **config)
        
    async def extract_elements(self, text: str) -> Dict[str, List[str]]:
        """Use LLM to extract key elements from text."""
        prompt = SUMMARIZE_PROMPT.format(text=text)
        
        responses = await generate_responses(
            self.agent,
            [prompt],
            system_message=SUMMARIZER_SYSTEM_PROMPT,
            K=1
        )
        
        # Parse the structured response
        response = responses[0][0]
        elements = {
            'objects': [],
            'actions': [],
            'locations': [],
            'changes': []
        }
        
        current_key = None
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Objects:'):
                current_key = 'objects'
                elements[current_key] = [x.strip() for x in line[8:].split(',') if x.strip()]
            elif line.startswith('Actions:'):
                current_key = 'actions'
                elements[current_key] = [x.strip() for x in line[8:].split(',') if x.strip()]
            elif line.startswith('Locations:'):
                current_key = 'locations'
                elements[current_key] = [x.strip() for x in line[10:].split(',') if x.strip()]
            elif line.startswith('Changes:'):
                current_key = 'changes'
                elements[current_key] = [line[8:].strip()]
                
        return elements

    async def summarize_interaction(
        self,
        current_state: str,
        action: str,
        next_state: str
    ) -> str:
        """Create a concise summary of an interaction cycle using LLM extraction."""
        # Extract elements from each component
        current_elements = await self.extract_elements(current_state)
        action_elements = await self.extract_elements(action)
        next_elements = await self.extract_elements(next_state)
        
        # Build concise summary
        summary_parts = []
        
        # Summarize state
        if current_elements['objects']:
            summary_parts.append(f"State: {', '.join(current_elements['objects'])}")
        
        # Summarize action
        if action_elements['actions']:
            action_summary = f"Action: {action_elements['actions'][0]}"
            if action_elements['objects']:
                action_summary += f" {action_elements['objects'][0]}"
            summary_parts.append(action_summary)
        
        # Summarize changes
        if next_elements['changes']:
            summary_parts.append(f"Result: {next_elements['changes'][0]}")
            
        return ' | '.join(summary_parts)

# Example usage:
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    
    async def test_summarizer():
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        summarizer = SummarizerAgent(api_key=api_key)
        
        current = "A leather-bound tome sits on a wooden pedestal. Dust covers the shelves."
        action = "The agent carefully opens the tome on the pedestal."
        next_state = "The tome now lies open, revealing ancient text. The pedestal remains steady."
        
        summary = await summarizer.summarize_interaction(current, action, next_state)
        print(summary)
    
    asyncio.run(test_summarizer()) 