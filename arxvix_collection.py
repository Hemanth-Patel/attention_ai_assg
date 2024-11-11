import arxiv
from pydantic import BaseModel
from typing import List
import csv

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

def save_papers_to_csv(papers: List[ResearchPaper], filename: str = "research_papers.csv"):
    # Define CSV header
    fieldnames = ["title", "authors", "summary", "published_date", "link"]
    
    # Open the CSV file for writing
    with open(filename, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write each paper's data as a row in the CSV file
        for paper in papers:
            writer.writerow({
                "title": paper.title,
                "authors": ', '.join(paper.authors),  # Join authors into a single string
                "summary": paper.summary,
                "published_date": paper.published_date,
                "link": paper.link
            })
    print(f"Data saved to {filename}")

# Usage example
papers = search_papers(topic="pinn", max_results=5)
save_papers_to_csv(papers, filename="pinn_research_papers.csv")