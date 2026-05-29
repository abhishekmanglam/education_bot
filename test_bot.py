# test_bot.py — zero langchain.chains dependency, works on all versions

import warnings
warnings.filterwarnings("ignore")

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── STEP 1: Create curriculum file ───────────────────────────────────────
curriculum_text = """
Photosynthesis is the process by which plants convert sunlight into food.
Plants use chlorophyll to absorb light energy from the sun.
The light energy splits water molecules into hydrogen and oxygen.
Oxygen is released as a byproduct — this is the oxygen we breathe.
Carbon dioxide from the air combines with hydrogen to form glucose.
Glucose is the sugar that gives the plant energy to grow.

The equation for photosynthesis is:
6CO2 + 6H2O + light energy → C6H12O6 + 6O2

Photosynthesis happens in two stages: light-dependent reactions and
the Calvin cycle. Light-dependent reactions occur in the thylakoid
membranes. The Calvin cycle occurs in the stroma of the chloroplast.

Cell division is the process by which cells reproduce.
Mitosis produces two identical daughter cells from one parent cell.
Mitosis has four phases: prophase, metaphase, anaphase, and telophase.
During prophase, chromosomes condense and become visible.
During metaphase, chromosomes line up along the cell equator.
During anaphase, chromosomes are pulled to opposite poles.
During telophase, two new nuclei form and the cell divides.
"""

with open("curriculum.txt", "w", encoding="utf-8") as f:
    f.write(curriculum_text)
print("✓ Step 1: Curriculum file created")


# ── STEP 2: Load and chunk ────────────────────────────────────────────────
loader = TextLoader("curriculum.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)

print(f"✓ Step 2: Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: '{chunk.page_content[:60]}...'")


# ── STEP 3: Embed and store ───────────────────────────────────────────────
print("\n⏳ Step 3: Embedding chunks (30-60 seconds first time)...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./test_chroma_db"
)
print("✓ Step 3: Chunks embedded and stored")


# ── STEP 4: Test retrieval ────────────────────────────────────────────────
print("\n🔍 Step 4: Testing retrieval...")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

test_query = "How does photosynthesis work?"
results = retriever.invoke(test_query)

print(f"Query: '{test_query}'")
print(f"Retrieved {len(results)} chunks:")
for i, doc in enumerate(results):
    print(f"\n  Chunk {i+1}:\n  {doc.page_content}")


# ── STEP 5: Build the bot (no langchain.chains needed) ───────────────────
print("\n🤖 Step 5: Building tutoring bot...")

llm = ChatOllama(model="llama3.2", temperature=0.3)

def format_docs(docs):
    """
    Takes a list of retrieved Document objects
    and joins their text into one string.
    This becomes the {context} in the prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(user_input: str, chat_history: list) -> str:
    """
    Full RAG pipeline manually wired:
    1. Retrieve relevant chunks for user_input
    2. Format chunks into context string
    3. Build prompt with context + history + question
    4. Send to LLM
    5. Return answer string
    """

    # Step A: retrieve relevant chunks
    retrieved_docs = retriever.invoke(user_input)
    # retriever.invoke() embeds user_input as a vector
    # finds top-2 most similar chunks from Chroma
    # returns list of Document objects

    context = format_docs(retrieved_docs)
    # joins chunk texts into one block
    # e.g.:
    # "Photosynthesis is the process...
    #  Plants use chlorophyll..."

    # Step B: build the prompt manually
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a patient and encouraging tutor.
Answer the student's question using ONLY the context provided below.
If the answer is not in the context, say:
'I don't have that in the curriculum.'
Keep explanations clear and simple.
After answering, ask one checking question to verify understanding.

Context:
{context}"""),

        MessagesPlaceholder(variable_name="chat_history"),
        # Filled with HumanMessage/AIMessage objects from chat_history list
        # Gives the LLM memory of previous turns

        ("human", "{input}"),
        # The student's current question
    ])

    # Step C: format the prompt with actual values
    formatted_prompt = prompt.format_messages(
        context=context,
        chat_history=chat_history,
        input=user_input
    )
    # formatted_prompt is now a list of messages ready to send to the LLM
    # e.g.:
    # [SystemMessage(content="You are a tutor... Context: Photosynthesis is..."),
    #  HumanMessage(content="What is photosynthesis?")]

    # Step D: send to LLM and get response
    response = llm.invoke(formatted_prompt)
    # llm.invoke() sends messages to local llama3.2
    # returns an AIMessage object with .content attribute

    return response.content
    # .content is the actual text string of the answer

print("✓ Bot ready!\n")
print("=" * 50)
print("TUTORING BOT — type 'quit' to exit")
print("=" * 50)


# ── STEP 6: Chat loop ─────────────────────────────────────────────────────
chat_history = []
# Starts empty — no memory at beginning of session
# We manually append to this after each turn

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Session ended.")
        break

    if not user_input:
        continue

    print("Bot: thinking...")

    answer = get_answer(user_input, chat_history)
    # Runs the full RAG pipeline defined above

    print(f"\nBot: {answer}")

    # Save this turn to history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=answer))
    # Next call to get_answer() will include these in the prompt
    # so the bot remembers what was said

    # Debug info
    print(f"\n[Memory: {len(chat_history)} messages]")
    print(f"[History: {[type(m).__name__ for m in chat_history]}]")
    # Shows e.g.: [HumanMessage, AIMessage, HumanMessage, AIMessage]