# import arxiv
# from pydantic import BaseModel
# from typing import List
# import ollama
# from fastapi import FastAPI, Query
# from py2neo import Graph, Node, Relationship

# app = FastAPI()

# # Neo4j connection setup
# # neo4j_url = "http://localhost:8000"
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
#     response = ollama.chat(model="llama", messages=[{"role": "user", "content": prompt}], stream=True)
#     answer = ""
#     for chunk in response:
#         answer += chunk['message']['content']
#     return {"response": answer}



# import streamlit as st
# import requests

# st.title("Academic Research Paper Assistant")

# API_URL = "http://127.0.0.1:8000"

# # Search and Store Papers
# st.header("Search and Store Research Papers")
# topic = st.text_input("Enter Research Topic")
# max_results = st.number_input("Number of Papers", min_value=1, max_value=20, value=5)
# if st.button("Search and Store"):
#     response = requests.post(f"{API_URL}/search_and_store/", params={"topic": topic, "max_results": max_results})
#     st.write(response.json()["message"])

# # Get Papers by Topic and Year
# st.header("Retrieve Papers by Topic and Year")
# year = st.text_input("Enter Year")
# if st.button("Retrieve Papers"):
#     response = requests.get(f"{API_URL}/get_papers/", params={"topic": topic, "year": year})
#     papers = response.json()["papers"]
#     for paper in papers:
#         st.subheader(paper["title"])
#         st.write("Summary:", paper["summary"])
#         st.write("Link:", paper["link"])

# # Q&A Section
# st.header("Ask Questions About Papers")
# user_question = st.text_input("Enter your question")
# if st.button("Get Answer"):
#     # Retrieve the latest papers stored for context
#     response = requests.get(f"{API_URL}/get_papers/", params={"topic": topic, "year": year})
#     papers = response.json()["papers"]

#     # Prepare context from papers
#     context = "\n".join([f"{paper['title']}: {paper['summary']}" for paper in papers])
#     full_prompt = f"Context:\n{context}\n\nQuestion: {user_question}"
    
#     # Query the Llama model with the prepared prompt
#     response = requests.post(f"{API_URL}/query_llama/", json={"prompt": full_prompt})
#     st.write(response.json()["response"])




# import streamlit as st
# import requests
# import time

# # Set up Streamlit page
# st.set_page_config(page_title="Academic Research Paper Assistant", layout="wide")
# st.title("Academic Research Paper Assistant")

# # Define API URL
# API_URL = "http://127.0.0.1:8000"

# # Initialize session states for chat history and response state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "is_generating_response" not in st.session_state:
#     st.session_state.is_generating_response = False

# # Function to add messages to chat history
# def chat_actions(user_input, response):
#     # Add user input to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     # Add assistant response to chat history
#     st.session_state.chat_history.append({"role": "assistant", "content": response})

# # Function to handle intent classification and execute actions based on it
# def handle_user_input(user_input):
#     # Step 1: Use the Llama model to classify the intent
#     response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Classify the intent of this input: '{user_input}'"})
#     intent = response.json().get("response", "").lower()
#     print(intent)

#     # Step 2: Execute actions based on the classified intent
#     if "greeting" in intent:
#         # Respond conversationally for greetings
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Respond to a greeting: '{user_input}'"})
#         bot_response = response.json().get("response", "Hello! How can I assist you with research papers today?")
    
#     elif "request papers" in intent:
#         # Extract number of papers and topic
#         num_papers = 5  # default value
#         topic = "research"  # generic topic in case no specific one is found

#         # Use Llama model to parse the input and extract details
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Extract the topic and number of papers from the input: '{user_input}'"})
#         parsed_info = response.json().get("response", "")
        
#         # Parse the response to get topic and number of papers
#         for line in parsed_info.splitlines():
#             if "topic" in line.lower():
#                 topic = line.split(":")[-1].strip()
#             if "number of papers" in line.lower():
#                 num_papers = int(line.split(":")[-1].strip())

#         # Fetch papers from the backend
#         response = requests.get(f"{API_URL}/get_papers/", params={"topic": topic, "limit": num_papers})
#         papers = response.json().get("papers", [])

#         # Format response with list of papers
#         if papers:
#             bot_response = f"Here are the top {num_papers} papers on '{topic}':\n"
#             for i, paper in enumerate(papers, 1):
#                 bot_response += f"{i}. {paper['title']} - [Link]({paper['link']})\n"
#         else:
#             bot_response = f"No papers found on '{topic}'."

#     elif "details" in intent:
#         # Fetch details of a specific paper
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Extract the title of the paper from the input: '{user_input}'"})
#         title = response.json().get("response", "").strip()

#         # Fetch specific paper details from the backend
#         response = requests.get(f"{API_URL}/fetch_paper_details/", params={"title": title})
#         paper = response.json().get("paper", {})

#         # Format response with paper details
#         if paper:
#             bot_response = f"**Title**: {paper['title']}\n**Authors**: {', '.join(paper['authors'])}\n" \
#                            f"**Summary**: {paper['summary']}\n**Published Date**: {paper['published_date']}\n" \
#                            f"**Link**: {paper['link']}\n"
#         else:
#             bot_response = f"Could not find detailed information for '{title}'."

#     else:
#         # General response using Llama model
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Respond to this input: '{user_input}'"})
#         bot_response = response.json().get("response", "I'm sorry, I didn't understand that.")

#     return bot_response

# # Display chat messages in a threaded format
# def display_chat():
#     for message in st.session_state.chat_history:
#         if message["role"] == "user":
#             with st.chat_message(name="You"):
#                 st.write(message["content"])
#         else:
#             with st.chat_message(name="Bot"):
#                 st.write(message["content"])

# # Chat Input Interface
# prompt = st.chat_input("Enter your message")

# # Handle chat input
# if prompt and not st.session_state.is_generating_response:
#     # Add user input to chat history with "Waiting for response" status
#     st.session_state.chat_history.append({"role": "user", "content": prompt})
#     st.session_state.chat_history.append({"role": "assistant", "content": "Waiting for response..."})

#     # Set the is_generating_response flag to True to disable input
#     st.session_state.is_generating_response = True

#     # Process the user input and determine the response
#     bot_response = handle_user_input(prompt)

#     # Remove the "Waiting for response" status and update with actual response
#     st.session_state.chat_history.pop()
#     chat_actions(prompt, bot_response)

#     # Reset the is_generating_response flag to False after response is processed
#     st.session_state.is_generating_response = False

# # Display the conversation history in a chatbot-like format
# display_chat()





# import streamlit as st
# import requests

# # Set up Streamlit page
# st.set_page_config(page_title="Academic Research Paper Assistant", layout="wide")
# st.title("Academic Research Paper Assistant")

# # Define API URL
# API_URL = "http://127.0.0.1:8000"

# # Initialize session states for chat history, response state, and stored papers
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "is_generating_response" not in st.session_state:
#     st.session_state.is_generating_response = False
# if "stored_papers" not in st.session_state:
#     st.session_state.stored_papers = {}

# # Function to add messages to chat history
# def add_message(role, content):
#     st.session_state.chat_history.append({"role": role, "content": content})


# # Function to handle user input and generate appropriate responses
# def handle_user_input(user_input):
#     # Step 1: Use the Llama model to classify the intent
#     response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Classify the intent of this input: '{user_input}'"})
#     intent = response.json().get("response", "").lower()

#     # Step 2: Execute actions based on the classified intent
#     if "greeting" in intent:
#         # Respond conversationally for greetings
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Respond to a greeting: '{user_input}'"})
#         bot_response = response.json().get("response", "Hello! How can I assist you with research papers today?")
    
#     elif "request papers" in intent:
#         # Extract number of papers and topic
#         num_papers = 5  # default value
#         topic = "research"  # generic topic in case no specific one is found

#         # Use Llama model to parse the input and extract details
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Extract the topic and number of papers from the input: '{user_input}'"})
#         parsed_info = response.json().get("response", "")
        
#         # Parse the response to get topic and number of papers
#         for line in parsed_info.splitlines():
#             if "topic" in line.lower():
#                 topic = line.split(":")[-1].strip()
#             if "number of papers" in line.lower():
#                 num_papers = int(line.split(":")[-1].strip())

#         # Fetch papers from the backend
#         response = requests.get(f"{API_URL}/get_papers/", params={"topic": topic, "limit": num_papers})
#         papers = response.json().get("papers", [])

#         # Format response with list of papers
#         if papers:
#             PAPERS_DATA = papers
#             bot_response = f"Here are the top {num_papers} papers on '{topic}':\n"
#             for i, paper in enumerate(papers, 1):
#                 bot_response += f"{i}. {paper['title']} - [Link]({paper['link']})\n"
#                 # Store each paper's information in session state for reference
#                 st.session_state.stored_papers[paper['title'].lower()] = paper
#         else:
#             bot_response = f"No papers found on '{topic}'."

#     elif "details" in intent or "abstract" in intent:
#         # Fetch details of a specific paper
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Extract the title of the paper from the input: '{user_input}'"})
#         title = response.json().get("response", "").strip().lower()

#         # Check if the paper exists in the stored papers
#         if title in st.session_state.stored_papers:
#             paper = st.session_state.stored_papers[title]
#             # Format detailed response based on the content of the paper
#             bot_response = f"**Title**: {paper['title']}\n**Authors**: {', '.join(paper['authors'])}\n" \
#                            f"**Summary**: {paper['summary']}\n**Published Date**: {paper['published_date']}\n" \
#                            f"**Link**: {paper['link']}\n"
#         else:
#             # If the paper is not found, try fetching full details from the backend
#             response = requests.get(f"{API_URL}/fetch_paper_details/", params={"title": title})
#             paper = response.json().get("paper", {})

#             if paper:
#                 # Store the detailed paper information
#                 st.session_state.stored_papers[paper['title'].lower()] = paper
#                 bot_response = f"**Title**: {paper['title']}\n**Authors**: {', '.join(paper['authors'])}\n" \
#                                f"**Summary**: {paper['summary']}\n**Published Date**: {paper['published_date']}\n" \
#                                f"**Link**: {paper['link']}\n"
#             else:
#                 bot_response = f"Could not find detailed information for '{title}'."

#     elif "follow-up question" in intent or "q/a" in intent or 'information' in intent or 'classification' in intent:
#         # Use the context of previously discussed papers for Q&A
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Generate a response based on previous papers for the question: '{user_input}' and context: {st.session_state.stored_papers}"})
#         bot_response = response.json().get("response", "I'm sorry, I didn't understand that.")

#     else:
#         # General response using Llama model
#         response = requests.post(f"{API_URL}/query_llama/", params={"prompt": f"Respond to this input: '{user_input}' and context: {st.session_state.stored_papers}"})
#         bot_response = response.json().get("response", "I'm sorry, I didn't understand that.")

#     return bot_response

# # Display chat messages in a threaded format
# def display_chat():
#     for message in st.session_state.chat_history:
#         if message["role"] == "user":
#             with st.chat_message(name="You"):
#                 st.write(message["content"])
#         else:
#             with st.chat_message(name="Bot"):
#                 st.write(message["content"])

# # Chat Input Interface
# prompt = st.chat_input("Enter your message")

# # Handle chat input
# if prompt and not st.session_state.is_generating_response:
#     # Add user input to chat history with "Waiting for response" status
#     st.session_state.chat_history.append({"role": "user", "content": prompt})
#     st.session_state.chat_history.append({"role": "assistant", "content": "Waiting for response..."})

#     # Set the is_generating_response flag to True to disable input
#     st.session_state.is_generating_response = True

#     # Process the user input and determine the response
#     bot_response = handle_user_input(prompt)

#     # Remove the "Waiting for response" status and update with actual response
#     st.session_state.chat_history.pop()
#     add_message("assistant", bot_response)

#     # Reset the is_generating_response flag to False after response is processed
#     st.session_state.is_generating_response = False

# # Display the conversation history in a chatbot-like format
# display_chat()







import streamlit as st
import requests

# # Set up Streamlit page
st.set_page_config(page_title="Academic Research Paper Assistant", layout="wide")
st.title("Academic Research Paper Assistant")

class AcademicResearchAssistant:
    def __init__(self, api_url="http://127.0.0.1:8000"):
        self.api_url = api_url
        self.initialize_session_state()
        self.paper_data = None

    def initialize_session_state(self):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "is_generating_response" not in st.session_state:
            st.session_state.is_generating_response = False
        if "stored_papers" not in st.session_state:
            st.session_state.stored_papers = {}

    def add_message(self, role, content):
        st.session_state.chat_history.append({"role": role, "content": content})

    def classify_intent(self, user_input):
        response = requests.post(f"{self.api_url}/query_llama/", params={"prompt": f"Classify the intent of this input: '{user_input}' among classes : ['greetings , request papers, details, follow-up question,general]"})
        return response.json().get("response", "").lower()

    def fetch_papers(self, topic, num_papers):
        response = requests.get(f"{self.api_url}/get_papers/", params={"topic": topic, "limit": num_papers})
        return response.json().get("papers", [])

    def fetch_paper_details(self, title):
        response = requests.get(f"{self.api_url}/fetch_paper_details/", params={"title": title})
        return response.json().get("paper", {})

    def handle_user_input(self, user_input):
        intent = self.classify_intent(user_input)

        if "greeting" in intent:
            print('inside greetings')
            response = requests.post(f"{self.api_url}/query_llama/", params={"prompt": f"Respond to a greeting: '{user_input}'"})
            return response.json().get("response", "Hello! How can I assist you with research papers today?")

        elif "request papers" in intent:
            print('inside req papers')
            return self.handle_request_papers(user_input)

        elif "details" in intent or "abstract" in intent:
            print('inside details')
            return self.handle_paper_details(user_input)

        elif "follow-up question" in intent or "q/a" in intent or 'information' in intent or 'classification' in intent:
            print('inside follow up')
            print('papers : ', self.paper_data)
            response = requests.post(f"{self.api_url}/query_llama/", params={"prompt": f"Generate a response based on previous papers for the question: '{user_input}' and papers : {self.paper_data}"})
            return response.json().get("response", "I'm sorry, I didn't understand that.")

        else:
            print('inside else')
            response = requests.post(f"{self.api_url}/query_llama/", params={"prompt": f"Respond to this input: '{user_input}' and papers : {self.paper_data}"})
            return response.json().get("response", "I'm sorry, I didn't understand that.")
        

    def handle_request_papers(self, user_input):
        response = requests.post(f"{self.api_url}/query_llama/", params={"prompt": f"Extract the topic and number of papers from the input: '{user_input}'"})
        parsed_info = response.json().get("response", "")

        num_papers = 5
        topic = "research"

        for line in parsed_info.splitlines():
            if "topic" in line.lower():
                topic = line.split(":")[-1].strip()
            if "number of papers" in line.lower():
                num_papers = int(line.split(":")[-1].strip())

        papers = self.fetch_papers(topic, num_papers)

        def format_papers(papers):
            formatted_papers = []
            for paper in papers:
                formatted_papers.append(
                    f"Title: {paper['title']}\n"
                    f"Authors: {', '.join(paper['authors'])}\n"
                    f"Summary: {paper['summary']}\n"
                    f"Published Date: {paper['published_date']}\n"
                    f"Link: {paper['link']}\n"
                    "----------------------------------------"
                )
            return "\n".join(formatted_papers)
        
        if papers:
            bot_response = f"Here are the top {num_papers} papers on '{topic}':\n"
            for i, paper in enumerate(papers, 1):
                bot_response += f"{i}. {paper['title']} - [Link]({paper['link']})\n"
                st.session_state.stored_papers[paper['title'].lower()] = paper
            self.paper_data = format_papers(papers)
            
            return bot_response
        else:
            return f"No papers found on '{topic}'."

    def handle_paper_details(self, user_input):
        print('paper ', self.paper_data)
        response = requests.post(f"{self.api_url}/query_llama/", params={"prompt": f"Extract the title of the paper from the input: '{user_input} and papers : {self.paper_data}'"})
        title = response.json().get("response", "").strip().lower()

        if title in st.session_state.stored_papers:
            paper = st.session_state.stored_papers[title]
            return f"**Title**: {paper['title']}\n**Authors**: {', '.join(paper['authors'])}\n" \
                   f"**Summary**: {paper['summary']}\n**Published Date**: {paper['published_date']}\n" \
                   f"**Link**: {paper['link']}\n"
        else:
            paper = self.fetch_paper_details(title)
            if paper:
                st.session_state.stored_papers[title] = paper
                return f"**Title**: {paper['title']}\n**Authors**: {', '.join(paper['authors'])}\n" \
                       f"**Summary**: {paper['summary']}\n**Published Date**: {paper['published_date']}\n" \
                       f"**Link**: {paper['link']}\n"
            else:
                return f"Could not find detailed information for '{title}'."

    def display_chat(self):
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message(name="You"):
                    st.write(message["content"])
            else:
                with st.chat_message(name="Bot"):
                    st.write(message["content"])

    def handle_chat(self):
        prompt = st.chat_input("Enter your message")

        if prompt and not st.session_state.is_generating_response:
            self.add_message("user", prompt)
            self.add_message("assistant", "Waiting for response...")

            st.session_state.is_generating_response = True

            bot_response = self.handle_user_input(prompt)

            st.session_state.chat_history.pop()  # Remove "Waiting for response"
            self.add_message("assistant", bot_response)

            st.session_state.is_generating_response = False

        self.display_chat()


# Create an instance of the assistant and start the chat handling
assistant = AcademicResearchAssistant()
assistant.handle_chat()
