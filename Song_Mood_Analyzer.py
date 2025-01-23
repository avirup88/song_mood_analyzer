import streamlit as st
import os
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
import traceback as tb
from datetime import datetime as dt
import chromadb
from langchain_community.vectorstores import Chroma

# Helper function to format document prompts
def fn_get_document_prompt(docs):
    """
    Generate a formatted string containing content from retrieved documents.
    Args:
        docs (list): List of documents retrieved from the database.
    Returns:
        str: Formatted document content.
    """
    prompt = "\n"
    for doc in docs:
        prompt += "\nContent:\n"
        prompt += doc.page_content + "\n\n"
    return prompt

# Streamlit App
st.title("🎵 Song Mood Analyzer")
st.write("A chatbot to analyze the mood of song lyrics and provide detailed insights.")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

@st.cache_data
def load_and_split_document(filename):
    """
    Load and split a PDF document into manageable chunks.
    """
    document_dir = "./"
    file_path = os.path.join(document_dir, filename)

    # Load and split the document
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    # Split pages into chunks
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    chunks = text_splitter.split_documents(pages)
    return chunks

@st.cache_resource
def setup_embeddings_and_db(_chunks):
    """
    Set up the embeddings and Chroma database for similarity search.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    client = chromadb.PersistentClient(path="./lyrics_db", settings=Settings(anonymized_telemetry=False))
    db = Chroma.from_documents(_chunks, embeddings, persist_directory="./lyrics_db", client=client)
    return db

def analyze_song_mood(question, filename="song_lyrics_book.pdf"):
    try:
        start_time = dt.now()
        st.info("🔍 Starting the mood analysis process...")

        # Load and split the document
        st.write("📄 Loading and splitting the lyrics document...")
        chunks = load_and_split_document(filename)

        # Setup Chroma database
        st.write("📂 Setting up the database for similarity search...")
        db = setup_embeddings_and_db(chunks)

        # Retrieve relevant documents
        st.write("🔄 Retrieving relevant documents based on your question...")
        retrieved_docs = db.similarity_search(question, k=5)

        # Generate formatted context
        formatted_context = fn_get_document_prompt(retrieved_docs)

        prompt = f"""
        ## SYSTEM ROLE
        You are a knowledgeable and creative chatbot designed to assist with mood detection about **Song Lyrics**, especially focusing on **Emotional Complexity**.

        ## USER QUESTION
        The user has asked: 
        "{question}"

        ## CONTEXT
        Here is the relevant content from the song lyrics book:  
        '''
        {formatted_context}
        '''

        ## GUIDELINES
        1. **Genius Lyrics Link Handling**:
        - If a Genius lyrics link is provided in the `CONTEXT`, validate it to ensure it follows the format `https://genius.com/ followed by the song name and artist appended with -lyrics in the URL`.
        - Use the validated link directly in the response if it exists.
        - Example format of the link https://genius.com/Billie-eilish-birds-of-a-feather-lyrics where Billie-eilish is the artist, birds-of-a-feather is the song followed by the -lyrics 
        - If no link is provided or the link is invalid, use the training data to get the Genius lyrics link of the song.

        2. **Validation**:
        - Ensure the Genius lyrics link format is correct before embedding it in the response.
        - Avoid defaulting to hardcoded or placeholder links.

        3. **Clarity**:
        - Use simple, professional, and concise language.  
        - Format your response in Markdown to ensure readability, particularly for Streamlit displays.  

        ## TASK
        1. Answer the user's question **directly** if possible.  
        2. Validate and embed the Genius lyrics link from the `CONTEXT` if it exists.
        3. If no valid link is present, use the training data to get the Genius lyrics link. 
        4. Adhere strictly to the response format provided below.

        ## RESPONSE FORMAT
        🎶 **Song Mood Analysis** 🎶

        - **Song:** Song Name
        - **Album:** Album Name
        - **Artist:** Artist Name
        - **Mood:** Mood in one word
        - **Genius Lyrics:** Genius Song Link
        - **Analysis:**
        - Bullet points explaining how the mood was determined, based on specific lines from the lyrics.
        """


        # Set up GPT client and parameters
        model_params = {
            'model': 'gpt-4',
            'temperature': 0.3,
            'max_tokens': 2000,
            'top_p': 0.95,
            'frequency_penalty': 0.8,
            'presence_penalty': 0.8
        }

        messages = [{'role': 'system', 'content': "You are a helpful assistant."},
                    {'role': 'user', 'content': prompt}]

        st.write("🤖 Querying the AI model for mood analysis...")
        completion = openai.OpenAI().chat.completions.create(messages=messages, **model_params, timeout=120)
        answer = completion.choices[0].message.content

        end_time = dt.now()
        duration = end_time - start_time
        td_mins = int(round(duration.total_seconds() / 60))
        st.success(f"✅ Analysis completed in {td_mins} minutes.")

        return answer

    except Exception as e:
        error = f"❌ Error occurred: {e}\nTraceback: {tb.format_exc()}"
        st.error(error)

# MAIN FUNCTION
if __name__ == '__main__':

    try:

        # User input
        st.write("🎯 **Select a prompt to analyze the mood of a song:**")
        prompt_options = [
            "What is the mood of the song Birds of a Feather by Billie Eilish?",
            "What emotions are portrayed in the song Hallelujah by Leonard Cohen?",
            "How does the song Someone Like You by Adele make you feel?",
            "How does the song Merry Christmas Everyone by Shakin' Stevens convey the spirit of celebration?",
            "What festive emotions are captured in All I Want for Christmas Is You by Mariah Carey?",
            "What mood is expressed in Underneath the Tree by Kelly Clarkson?",
            "How does Santa Tell Me by Ariana Grande reflect the emotions of love and uncertainty during Christmas?",
            "What emotions dominate Sailor Song by Gigi Perez?",
            "How does Snowman by Sia encapsulate winter themes and emotions?",
            "What mood is created by It's Beginning to Look a Lot like Christmas by Michael Bublé?",
            "How does Unwritten by Natasha Bedingfield inspire feelings of new beginnings?",
            "What emotional depth is found in Wildflower by Billie Eilish?",
            "How does Apocalypse by Cigarettes After Sex portray longing and melancholy?",
            "What emotions resonate in Yellow by Coldplay?",
            "How does Hallelujah by Leonard Cohen blend spirituality with human emotions?",
            "What is the mood of Another Love by Tom Odell, and how does it explore heartache?",
            "How does Flowers by Miley Cyrus express themes of self-love and independence?",
            "What emotions are conveyed in Sweater Weather by The Neighbourhood?",
            "What mood is evoked in Locked Out of Heaven by Bruno Mars?",
            "How does Wake Me Up by Avicii inspire a sense of journey and self-discovery?",
            "What emotional landscape is created in Forever Young by Alphaville?",
            "How does Stargazing by Myles Smith depict themes of love and wonder?"
        ]


        selected_prompt = st.selectbox("Select a prompt:", prompt_options)

        col1, col2 = st.columns([4, 1])

        with col1:
            if st.button("Submit"):
                with st.spinner("⏳ Analyzing song mood..."):
                    result = analyze_song_mood(selected_prompt)
                    st.markdown("### 🎵 **Mood Analysis Result**")
                    st.markdown(result, unsafe_allow_html=True)

        with col2:
            if st.button("Reset"):
                st.rerun()

    except Exception as e:
        error = "Mood Analyzer Bot Failure :: Error Message {} with Traceback Details: {}".format(e, tb.format_exc())        
        print(error)
