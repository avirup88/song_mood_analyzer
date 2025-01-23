# 🎵 Song Mood Analyzer

## Overview
**Song Mood Analyzer** is a Streamlit-based web application that analyzes the mood and emotional depth of song lyrics. 
The app uses advanced AI models to extract insights from lyrics, identify emotional tones, and provide detailed mood analysis, 
along with validated Genius links for the lyrics.

---

## Features
- **AI-powered Mood Analysis:** Leverages OpenAI's GPT model to analyze song lyrics.
- **Document Management:** Loads and processes PDF lyric books with efficient chunking.
- **Similarity Search:** Utilizes vector embeddings and a Chroma database to match user queries to song lyrics.
- **Streamlit Integration:** Provides a user-friendly interface for mood analysis and detailed results.
- **Genius Lyrics Validation:** Ensures links to lyrics are correct and follow a standardized format.

---

## Installation

### Prerequisites
- Python 3.8+
- A valid OpenAI API key
- Required Python packages listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/song-mood-analyzer.git
   cd song-mood-analyzer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables by creating a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run the app:
   ```bash
   streamlit run Song_Mood_Analyzer.py
   ```

---

## Usage
1. Upload a PDF file containing song lyrics.
2. Select a predefined question or input a custom one to analyze.
3. Click **Submit** to retrieve the mood analysis results.
4. View detailed insights and validated Genius links for the analyzed songs.

---

## Technologies Used
- **Streamlit:** For the interactive web interface.
- **OpenAI GPT Models:** For natural language processing and mood analysis.
- **ChromaDB:** To perform similarity search on song lyrics.
- **LangChain:** For managing document splitting and processing.
- **PyPDFLoader:** For loading and splitting PDF documents.
- **Python-dotenv:** To manage environment variables.

---

## File Structure
```
📂 song-mood-analyzer
├── Song_Mood_Analyzer.py       # Main application script
├── requirements.txt           # Required Python packages
├── .env                       # Environment variables file
├── README.md                  # Project documentation
└── lyrics_db/                 # Directory for storing Chroma database files
```


