from pydantic import BaseModel, Field
from typing import List, Optional
import ollama
import time
import json

class BiasDetail(BaseModel):
    bias_type: str = Field(description="The type of bias identified (e.g., Framing, Omission, Loaded Language).")
    explanation: str = Field(description="Explanation of why this is considered a bias in context.")
    evidence: str = Field(description="Specific quote or phrase from the article serving as evidence for the bias.")

class ArticleAnalysis(BaseModel):
    source_name: str = Field(description="The name of the news source for this article.")
    is_relevant: bool = Field(description="Whether the article was deemed relevant to the core news event.")
    relevance_reason: Optional[str] = Field(default=None, description="Reason if the article was deemed not relevant. Null if relevant.")
    identified_biases: List[BiasDetail] = Field(default_factory=list, description="List of biases identified in this article. Empty if no biases found or article irrelevant for bias analysis.")

class NewsOutput(BaseModel):
    news_title: str = Field(description="A  title of the summarized core news event identified from the relevant articles.")
    synthesized_neutral_report: str = Field(description="The synthesized, neutral news report based on relevant articles.")
    article_analyses: List[ArticleAnalysis] = Field(description="Analysis for each provided article source, including relevance and biases.")


# --- System Instruction ---
SYSTEM_INSTRUCTION = """
You are an expert, impartial journalist and media analyst. Your primary task is to analyze a collection of news articles and produce a structured JSON output.

Follow these steps meticulously:
1.  **Identify Core Event:** Based on ALL provided "INPUT ARTICLE #N" blocks, determine and briefly state the main news event or topic in the 'news_title' field.
2.  **Process Each Input Article Individually for Analysis:** For EACH "INPUT ARTICLE #N" provided (e.g., "INPUT ARTICLE #1", "INPUT ARTICLE #2", etc.):
    a.  Extract its 'source_name' from the "SOURCE_NAME_FOR_ARTICLE_#N" field associated with that input block.
    b.  **Relevance Check:** Assess if "INPUT ARTICLE #N" primarily discusses the identified core event. Set 'is_relevant' to true or false. If false, provide a brief 'relevance_reason'.
    c.  **Bias Analysis:** Using the "CONTENT_OF_ARTICLE_#N", identify potential biases (e.g., Framing, Omission, Loaded Language, Spin). For each bias, provide 'bias_type', 'explanation', and 'evidence' (a direct quote from "CONTENT_OF_ARTICLE_#N"). Populate the 'identified_biases' list for this article.
    d.  Construct an analysis object for THIS "INPUT ARTICLE #N" and add it to the 'article_analyses' list in the JSON output. Ensure the 'source_name' in this JSON object matches the source from "SOURCE_NAME_FOR_ARTICLE_#N".
3.  **Synthesize Neutral Report (from relevant articles ONLY):**
    a.  Using ONLY the content from "INPUT ARTICLE #N" blocks that were marked as 'is_relevant', create a 'synthesized_neutral_report'.
    b.  Focus strictly on verifiable facts, key events, figures, and attributed statements.
    c.  Present information in a balanced way.
    d.  If conflicting facts arise between relevant articles, state them (e.g., "Content from INPUT ARTICLE #1 reports X, while content from INPUT ARTICLE #2 reports Y.") without resolving them unless a resolution is present in the articles.
    e.  The tone must be objective and informative. Avoid speculation or emotionally charged language.

You MUST output your findings in the structured JSON format defined by the Pydantic schema.
The 'article_analyses' list MUST contain one distinct analysis object for EACH "INPUT ARTICLE #N" provided. For example, if two articles ("INPUT ARTICLE #1" and "INPUT ARTICLE #2") are provided, the 'article_analyses' list must contain exactly two objects, the first corresponding to #1 and the second to #2. Do NOT duplicate analysis for the same input block.
"""

def format_articles_for_prompt(articles_with_sources: list[dict]) -> str:
    prompt_text = "--- START OF PROVIDED ARTICLES TO ANALYZE ---\n\n"
    for i, item in enumerate(articles_with_sources):
        # Use a very distinct and numbered identifier for each article block
        prompt_text += f"--- INPUT ARTICLE #{i+1} ---\n"
        prompt_text += f"SOURCE_NAME_FOR_ARTICLE_#{i+1}: {item['source_name']}\n"
        prompt_text += f"CONTENT_OF_ARTICLE_#{i+1}:\n{item['content']}\n"
        prompt_text += f"--- END OF INPUT ARTICLE #{i+1} ---\n\n"
    prompt_text += "--- END OF PROVIDED ARTICLES TO ANALYZE ---"
    return prompt_text

start = time.time()

user_prompt_content = format_articles_for_prompt([
        {"source_name": "Al Jazeera", "content": open("demoNews/pahalgam_alzajeera.txt", "r", encoding="utf-8").read()},
        {"source_name": "OpIndia", "content": open("demoNews/pahalgam_opindia.txt", "r", encoding="utf-8").read()},
    ])

source_names_list = [
    "Al Jazeera",
    "OpIndia",
]
num_articles = len(source_names_list)
full_user_prompt = f"""
You will be provided with {num_articles} input articles, clearly demarcated as "INPUT ARTICLE #1", "INPUT ARTICLE #2", etc.
The sources are: {', '.join(source_names_list)}.

Your task is to meticulously follow the system instructions to analyze EACH of these {num_articles} input articles and generate the specified JSON output.
Ensure the 'article_analyses' list in your JSON output contains exactly {num_articles} entries, one for each correspondingly numbered input article.

{user_prompt_content}

Remember to produce a valid JSON object matching the schema provided in the system instructions.
"""

response = ollama.generate(
    options={
        "temperature": 0.3,
        "system": SYSTEM_INSTRUCTION,
    },
    prompt=full_user_prompt,  model='granite3.3:latest',  format=NewsOutput.model_json_schema(),
)

# print(response)

f = open("aggrResult.json", "w", encoding="utf-8")
f.write(response.response)
f.close()

end = time.time()

print("Time taken: ", end - start)