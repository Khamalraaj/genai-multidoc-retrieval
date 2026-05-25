## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Design an AI agent using LlamaIndex and OpenAI GPT-3.5 Turbo to answer questions from multiple research papers. The system should download papers, create document tools, build an indexed retriever, and use a function-calling agent to query and compare information from the papers Self-RAG, SWE-Bench, and LongLoRA.

### DESIGN STEPS:

#### STEP 1:
Download the research papers (PDF files) from the given URLs.
#### STEP 2:
Create tools for each paper to enable searching and summarizing the content.
#### STEP 3:

Build an AI agent, connect it with the tools, and ask questions to get summaries and comparisons from the papers.

### PROGRAM:

```
# STEP 1 — INSTALL REQUIRED LIBRARIES

import sys

!{sys.executable} -m pip install llama-index
!{sys.executable} -m pip install llama-index-llms-openai
!{sys.executable} -m pip install pypdf
!{sys.executable} -m pip install openai==0.28.1
!{sys.executable} -m pip install python-dotenv
!{sys.executable} -m pip install nest_asyncio

# STEP 2 — IMPORT REQUIRED LIBRARIES

import os
import openai
import nest_asyncio
import urllib.request

from dotenv import load_dotenv, find_dotenv

nest_asyncio.apply()
# STEP 3 — LOAD API KEY

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

print("API Loaded Successfully")

# STEP 4 — SELECT RESEARCH PAPERS

urls = [
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=yV6fD7LYkF",
    "https://openreview.net/pdf?id=hnrB5YHoYu",
    "https://openreview.net/pdf?id=WbWtOYIzIK"
]

papers = [
    "loftq.pdf",
    "swebench.pdf",
    "values.pdf",
    "finetune_fair_diffusion.pdf",
    "knowledge_card.pdf"
]

# STEP 5 — DOWNLOAD PAPERS (Reliable Version)

import urllib.request
import time

for url, paper in zip(urls, papers):

    success = False
    retries = 3

    for attempt in range(retries):

        try:
            print(f"Downloading {paper}... Attempt {attempt+1}")

            urllib.request.urlretrieve(url, paper)

            print(f"Successfully downloaded {paper}")
            success = True
            break

        except Exception as e:
            print(f"Error downloading {paper}: {e}")

            if attempt < retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)

    if not success:
        print(f"Failed to download {paper}")

print("Download Completed")
# STEP 6 — VERIFY DOWNLOADED FILES

import os

print(os.listdir())
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader
from llama_index.core.objects import ObjectIndex

from llama_index.llms.openai import OpenAI

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

paper_to_tools_dict = {}

for paper in papers:

    print(f"Processing {paper}")

    documents = SimpleDirectoryReader(
        input_files=[paper]
    ).load_data()

    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=f"{Path(paper).stem}_summary",
        description=f"Provides summary of {paper}"
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=f"{Path(paper).stem}_vector",
        description=f"Answers detailed questions from {paper}"
    )

    paper_to_tools_dict[paper] = [
        vector_tool,
        summary_tool
    ]
all_tools = [
    t for paper in papers
    for t in paper_to_tools_dict[paper]
]

print(len(all_tools))

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(
    similarity_top_k=3
)
llm = OpenAI(
    model="gpt-3.5-turbo"
)


agent_worker = FunctionCallingAgentWorker.from_tools(

    tool_retriever=obj_retriever,

    llm=llm,

    system_prompt="""
You are an agent designed to answer queries
over multiple research papers.

Always use the provided tools.

Do not rely on prior knowledge.
""",

    verbose=True
)

agent = AgentRunner(agent_worker)

response = agent.query(
    "What problem does LoftQ solve?"
)

print(response)

 
response = agent.query(
    "Compare SWEBench and LoftQ papers"
)
print("Name: Khamalraaj S , Rollno: 212224230122")
print(response)

response = agent.query(
    "Which paper discusses fairness in AI systems?"
)

print(response)
```

### OUTPUT:
<img width="1248" height="379" alt="image" src="https://github.com/user-attachments/assets/9edff959-fe87-40b7-952d-0643d116db2d" />
<img width="1076" height="816" alt="image" src="https://github.com/user-attachments/assets/3e7f2462-4856-4051-b827-32ff2b76bf48" />
<img width="1074" height="500" alt="image" src="https://github.com/user-attachments/assets/24f16c99-87e4-4894-97dc-8ed9a3ffa44c" />


### RESULT:

Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex is done Successfully
