import arxiv
from pydantic import BaseModel
from typing import List
import ollama

# Data model for a research paper
class ResearchPaper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    published_date: str
    link: str

def search_papers(topic: str, max_results: int = 5):
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        paper = ResearchPaper(
            title=result.title,
            authors=[author.name for author in result.authors],
            summary=result.summary,
            published_date=result.published.date().isoformat(),
            link=result.pdf_url
        )
        papers.append(paper)
    return papers

def format_papers_for_prompt(papers: List[ResearchPaper]) -> str:
    """
    Formats the list of ResearchPaper objects into a string for the Llama model prompt.
    """
    formatted_papers = []
    for paper in papers:
        formatted_papers.append(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Summary: {paper.summary}\n"
            f"Published Date: {paper.published_date}\n"
            f"Link: {paper.link}\n"
            "----------------------------------------"
        )
    return "\n".join(formatted_papers)

def query_llama_with_prompt(prompt: str, model="llama3"):
    """
    Sends a prompt to the Llama model available through Ollama and returns the response.
    """
    try:
        # response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        # return response['message']['content']

        stream = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,)

        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        return 'Done'
    except Exception as e:
        print(f"Error calling Ollama model: {e}")
        return None

# Main function to perform search, format results, and get Llama model response
def main(topic: str, max_results: int, user_prompt: str):
    # Step 1: Search for papers
    papers = search_papers(topic=topic, max_results=max_results)
    
    # Step 2: Format the papers for the prompt
    context = format_papers_for_prompt(papers)
    
    # Step 3: Combine the context with the user prompt
    full_prompt = f"{context}\n\nUser Question: {user_prompt}"
    
    # Step 4: Query the Llama model with the prompt
    response = query_llama_with_prompt(full_prompt, model="llama3")
    
    # Display the response
    print("Response from Llama Model:\n", response)

# Example usage
topic = "pinn"
max_results = 3
# user_prompt = "Summarize the main findings and recent advancements in the pinn."
user_prompt = ' what is pinn?'

main(topic, max_results, user_prompt)