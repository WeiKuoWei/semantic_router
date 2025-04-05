# Hierarchical Centroid-Based Router for Multi-Expert AI Systems

A scalable and efficient architecture for routing user queries to the appropriate domain expert in multi-expert systems, now with a modern web interface.

## Key Features

- **Two-Layer Semantic Routing**: Routes queries first to a general knowledge domain, then to a specific expert within that domain
- **Automatic Centroid Calculation**: Computes centroid vector representations of expert knowledge from PDF documents
- **Incremental Updates**: Only processes new files when updating centroids, tracks previously processed files
- **Weighted Centroids**: Handles different volumes of content per expert by using weighted averages
- **No Manual Utterances**: Eliminates the need to manually create utterances for each expert
- **Vector Database Integration**: Stores document embeddings in ChromaDB for efficient retrieval
- **LLM-Powered Responses**: Uses OpenAI's GPT models to generate responses based on retrieved content
- **Modern Web Interface**: React-based UI with support for markdown, code highlighting, and mathematical formulas

## Project Structure

```
project/
├── backend/
│   ├── tracking/
│   │   ├── processed_files.json
│   │   └── conversations/
│   ├── router/
│   │   ├── multi_layer_router.py
│   │   └── centroid_vectors.py
│   ├── utils/
│   │   ├── converter.py
│   │   ├── visualizer.py
│   │   ├── session_manager.py
│   │   ├── llm_service.py
│   │   ├── chromadb_handler.py
│   │   └── config.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   └── models/
│   ├── db/
│   └── main.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   └── ...
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   └── index.jsx
│   └── package.json
├── data/
│   ├── course_material/
│   │   ├── applied_machine_learning/
│   │   └── reinforcement_learning/
│   └── health_expert/
│       ├── mental_health/
│       └── physical_health/
├── .env
└── README.md
```

## How It Works

1. **Hierarchical Knowledge Organization**:
   - Knowledge is organized in a two-tier hierarchy (groups → experts → documents)
   - Each group contains multiple expert directories, each expert directory contains PDF files
   - This structure enables O(log n) routing complexity instead of O(n)

2. **Vector Representation**:
   - Each PDF is split into chunks and converted to embeddings
   - Each expert's knowledge is represented by a centroid vector (average of its document embeddings)
   - Each group's knowledge is represented by a weighted centroid of its experts' centroids
   - Individual document chunks are stored in ChromaDB for retrieval

3. **Query Routing Pipeline**:
   - When a query arrives, it's converted to an embedding vector
   - The system first finds the most similar group using cosine similarity
   - Then finds the most similar expert within that group
   - The system retrieves the most relevant document chunks from ChromaDB for the selected expert
   - These chunks and the query are sent to the OpenAI API
   - The LLM generates a comprehensive response based on the retrieved knowledge

4. **Web Interface**:
   - React frontend communicates with the FastAPI backend
   - Chat interface supports markdown, syntax highlighting, and math expressions
   - Session management for continuing conversations
   - Responsive design for desktop and mobile use

## Technical Advantages

- **Computational Efficiency**: Reduces routing decisions from O(n) to O(log n) complexity
- **Automatic Adaptation**: Centroids automatically update as knowledge bases evolve
- **Zero Manual Configuration**: No need for manual utterances or labeled training data
- **Incremental Learning**: New documents automatically update centroids without full recomputation
- **Collaborative Learning**: Multiple users can share and benefit from the same knowledge base
- **Enhanced User Experience**: Rich text rendering with support for code and mathematical formulas

## Setup

### Backend Setup

1. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   DB_PATH="../database"
   ```

2. Install backend requirements:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. In the data directory, place PDF files in the appropriate expert directories:
   - Each PDF should contain domain-specific knowledge
   - File names must be unique within each expert directory

### Frontend Setup

1. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Usage

### Data Processing

Process PDF files and generate centroids:
```bash
# launch backend and execute the process function at http://localhost:8000/docs, or
curl -X POST http://localhost:8000/api/process_file

```

This scans all PDF files, calculates embeddings, stores them in ChromaDB, and updates centroid vectors for all experts and groups.

### Running the Application

1. Start the backend API:
   ```bash
   cd backend
   python -m uvicorn api.main:app --reload
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

3. Open your browser to the URL shown in the Vite output (typically http://localhost:5173)

## Future Work

1. **Enhanced LLM Integration**:
   - Add support for different LLM providers (Anthropic, Llama, etc.)
   - Implement domain-specific prompt engineering for each expert
   - Add fine-tuning capabilities for expert-specific knowledge

2. **Dynamic Expert Discovery**:
   - Automatically detect new expert directories and update the routing system
   - Enable real-time updates to centroids as new content is added

3. **Frontend Enhancements**:
   - Add user authentication and user-specific data
   - Implement expert switching for specialized inquiries
   - Add visualization of document sources and relevance scores