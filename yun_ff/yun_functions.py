## GENERAL IMPORTS 
from difflib import get_close_matches
import os
import openai
from openai import OpenAI
import sys
sys.path.append("/Users/kaylahuang/Desktop/GitHub/conflicts/components/")

## MY IMPORTS
import enums 

# CONSTANTS
openai.organization = "org-raWgaVqCbuR9YlP1CIjclYHk" # Harvard
openai.api_key = os.getenv("OPENAI_API_KEY")


## FUNCTIONS
def fuzzy_match(input):
    """
    input: string given by user in search
    """
    closest_match = get_close_matches(input, enums.supported_countries, n=1, cutoff=0.6)
    return closest_match[0] if closest_match else None


## CLASSES

def llm_country_info(country, more_info=None):
    if more_info == '': more_info = None
    
    client = OpenAI()
    
    system_prompt = f"""You are foreign policy and political science expert advising a civil
        conflicts research team. You are asked to provide a brief summary of the history of 
        civil conflicts. When you are asked for a summary, you should provide a brief overview (LIMITED
        TO TWO PARAGRAPHS ONLY) of the timeline of important civil conflicts in the nation, as well 
        as the main belligerents. Do not focus too much on historical details and events."""

    completion = client.chat.completions.create(
        # turbo (1106-preview) has 128k context window, about 300 pages of text
        model="gpt-4-1106-preview", # test with: gpt-3.5-turbo, run final: gpt-4-1106-preview
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Tell me about the history of civil conflicts in {country}. Also, {more_info or ''} Limit your response to two short paragraphs not exceeding 200 words in total."},
        ]
    )

    gpt_summary = completion.choices[0].message.content
    return gpt_summary