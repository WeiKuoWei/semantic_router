# Hierarchical Centroid-Based Router for Multi-Expert AI Systems

A scalable and efficient architecture for routing user queries to the appropriate domain expert in multi-expert systems.

## Key Features

- **Two-Layer Semantic Routing**: Routes queries first to a general knowledge domain, then to a specific expert within that domain
- **Automatic Centroid Calculation**: Computes centroid vector representations of expert knowledge from PDF documents
- **Incremental Updates**: Only processes new files when updating centroids, tracks previously processed files
- **Weighted Centroids**: Handles different volumes of content per expert by using weighted averages
- **No Manual Utterances**: Eliminates the need to manually create utterances for each expert
- **Vector Database Integration**: Stores document embeddings in ChromaDB for efficient retrieval
- **LLM-Powered Responses**: Uses OpenAI's GPT models to generate responses based on retrieved content

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

## Technical Advantages

- **Computational Efficiency**: Reduces routing decisions from O(n) to O(log n) complexity
- **Automatic Adaptation**: Centroids automatically update as knowledge bases evolve
- **Zero Manual Configuration**: No need for manual utterances or labeled training data
- **Incremental Learning**: New documents automatically update centroids without full recomputation
- **Collaborative Learning**: Multiple users can share and benefit from the same knowledge base

## Setup

1. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   DB_PATH=src/db
   ```

2. Install requirements:
   ```bash
   pip install -r src/requirements.txt
   ```

3. Place PDF files in the appropriate expert directories:
   - Each PDF should contain domain-specific knowledge
   - File names must be unique within each expert directory
   - Example hierarchy with the provided data
     ```
     data/
     ├── course_material/
     │   ├── applied_machine_learning/
     │   │   └── [PDF files]
     │   └── reinforcement_learning/
     │       └── [PDF files]
     └── health_expert/
         ├── mental_health/
         │   └── [PDF files]
         └── physical_health/
             └── [PDF files]
     ```
   - Feel free to create more groups and expert

## Usage

1. **Process PDF files and generate centroids**:
   ```bash
   python src/main.py --process
   ```
   This scans all PDF files, calculates embeddings, stores them in ChromaDB, and updates centroid vectors for all experts and groups.

2. **Test with sample queries**:
   ```bash
   python src/main.py --test
   ```
   Tests the router with sample queries to see which expert handles each query.

3. **Interactive Query Mode**:
   ```bash
   python src/main.py --query
   ```
   Starts an interactive session where you can type queries and get responses from the appropriate expert.

## Future Work

1. **Improved Vector Search**:
   - Implement hybrid search combining semantic and keyword matching
   - Add support for filtering results by metadata
   - Optimize retrieval with approximate nearest neighbor search for larger collections

2. **Enhanced LLM Integration**:
   - Add support for different LLM providers (Anthropic, Llama, etc.)
   - Implement domain-specific prompt engineering for each expert
   - Add fine-tuning capabilities for expert-specific knowledge

3. **Dynamic Expert Discovery**:
   - Automatically detect new expert directories and update the routing system
   - Enable real-time updates to centroids as new content is added

4. **Performance Optimization**:
   - Implement caching for frequently asked queries
   - Optimize embedding generation for large document collections