# import arxiv
# from pydantic import BaseModel
# from typing import List
# import ollama
# from fastapi import FastAPI, Query
# from py2neo import Graph, Node, Relationship

# app = FastAPI()

# # Neo4j connection setup
# neo4j_url = "neo4j+s://bf0f24a2.databases.neo4j.io"
# neo4j_user = "neo4j"
# neo4j_password = "YJnU816NgPsf_MNkGQDJSdxaXviO-vOR8bjJfzOvd-k"
# graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))

# class ResearchPaper(BaseModel):
#     title: str
#     authors: List[str]
#     summary: str
#     published_date: str
#     link: str
#     topic: str

# def search_papers(topic: str, max_results: int = 5):
#     search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
#     papers = []
#     for result in search.results():
#         paper = ResearchPaper(
#             title=result.title,
#             authors=[author.name for author in result.authors],
#             summary=result.summary,
#             published_date=result.published.date().isoformat(),
#             link=result.pdf_url,
#             topic=topic
#         )
#         papers.append(paper)
#     return papers

# def store_papers_in_neo4j(papers: List[ResearchPaper]):
#     for paper in papers:
#         paper_node = Node("ResearchPaper", title=paper.title, summary=paper.summary,
#                           published_date=paper.published_date, link=paper.link, topic=paper.topic)
#         topic_node = Node("Topic", name=paper.topic)
#         graph.merge(topic_node, "Topic", "name")
#         graph.merge(paper_node, "ResearchPaper", "title")
#         graph.merge(Relationship(paper_node, "ABOUT", topic_node))

# def query_papers_by_topic_and_year(topic: str, year: str):
#     query = f"""MATCH (p:ResearchPaper)-[:ABOUT]->(t:Topic {{name: $topic}})
#                 WHERE p.published_date STARTS WITH $year
#                 RETURN p.title AS title, p.summary AS summary, p.link AS link"""
#     return graph.run(query, topic=topic, year=year).data()

# @app.post("/search_and_store/")
# async def search_and_store(topic: str, max_results: int = 5):
#     papers = search_papers(topic, max_results)
#     store_papers_in_neo4j(papers)
#     return {"message": f"Stored {len(papers)} papers on {topic}"}

# @app.get("/get_papers/")
# async def get_papers(topic: str, year: str = None):
#     papers = query_papers_by_topic_and_year(topic, year)
#     return {"papers": papers}

# @app.post("/query_llama/")
# async def query_llama(prompt: str):
#     response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}], stream=True)
#     answer = ""
#     for chunk in response:
#         answer += chunk['message']['content']
#     return {"response": answer}





import arxiv
from pydantic import BaseModel
from typing import List
import ollama
from fastapi import FastAPI, Query
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Neo4j connection setup with environment variables
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI="neo4j+s://bf0f24a2.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="YJnU816NgPsf_MNkGQDJSdxaXviO-vOR8bjJfzOvd-k"


# Connect to Neo4j Aura database
graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

class ResearchPaper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    published_date: str
    link: str
    topic: str

# Search Arxiv for papers related to a given topic
def search_papers(topic: str, max_results: int = 5):
    search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for result in search.results():
        paper = ResearchPaper(
            title=result.title,
            authors=[author.name for author in result.authors],
            summary=result.summary,
            published_date=result.published.date().isoformat(),
            link=result.pdf_url,
            topic=topic
        )
        papers.append(paper)
    return papers

# Store research papers in Neo4j
def store_papers_in_neo4j(papers: List[ResearchPaper]):
    for paper in papers:
        paper_node = Node("ResearchPaper", title=paper.title, summary=paper.summary,
                          published_date=paper.published_date, link=paper.link, topic=paper.topic)
        topic_node = Node("Topic", name=paper.topic)
        graph.merge(topic_node, "Topic", "name")
        graph.merge(paper_node, "ResearchPaper", "title")
        graph.merge(Relationship(paper_node, "ABOUT", topic_node))

# Query stored papers by topic and year
def query_papers_by_topic_and_year(topic: str, year: str):
    query = f"""MATCH (p:ResearchPaper)-[:ABOUT]->(t:Topic {{name: $topic}})
                WHERE p.published_date STARTS WITH $year
                RETURN p.title AS title, p.summary AS summary, p.link AS link"""
    return graph.run(query, topic=topic, year=year).data()

# Fetch detailed paper content by title from Arxiv
def fetch_paper_by_title(title: str):
    search = arxiv.Search(query=title, max_results=1)
    result = next(search.results(), None)
    if result:
        return {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published_date": result.published.date().isoformat(),
            "link": result.pdf_url
        }
    return None

@app.post("/search_and_store/")
async def search_and_store(topic: str, max_results: int = 5):
    papers = search_papers(topic, max_results)
    store_papers_in_neo4j(papers)
    return {"message": f"Stored {len(papers)} papers on {topic}"}

@app.get("/get_papers/")
async def get_papers(topic: str, year: str = None, limit: int = 5):
    papers = query_papers_by_topic_and_year(topic, year)[:limit]
    return {"papers": papers}

@app.post("/query_llama/")
async def query_llama(prompt: str):
    print('got request to llama3')
    stream = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}], stream=True)
    answer = ""
    for chunk in stream:
        answer += chunk['message']['content']
    print('answer: ', answer)
    return {"response": answer}

@app.get("/fetch_paper_details/")
async def fetch_paper_details(title: str):
    paper = fetch_paper_by_title(title)
    return {"paper": paper}
