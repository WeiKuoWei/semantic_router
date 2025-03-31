# Multi-Layer Semantic Router Prototype

A prototype implementation of a two-layer semantic router that intelligently routes queries to the appropriate expert based on their domain knowledge.

## Directory Structure

```
project/
├── data/
│   ├── course_material/
│   │   ├── expert_for_course_1/
│   │   │   └── [PDF files]
│   │   └── expert_for_course_2/
│   │       └── [PDF files]
│   └── health_expert/
│       ├── mental_health/
│       │   └── [PDF files]
│       └── physical_health/
│           └── [PDF files]
└── src/
    ├── tracking/
    │   └── processed_files.json
    ├── router/
    │   ├── multi_layer_router.py
    │   └── centroid_vectors.py
    ├── utils/
    │   ├── converter.py
    │   └── visualizer.py
    ├── main.py
    └── requirements.txt
```

## Features

- **Two-Layer Semantic Routing**: Routes queries first to a general knowledge domain, then to a specific expert within that domain
- **Automatic Centroid Calculation**: Computes centroid vector representations of expert knowledge from PDF documents
- **Incremental Updates**: Only processes new files when updating centroids, tracks previously processed files
- **Weighted Centroids**: Handles different volumes of content per expert by using weighted averages
- **No Manual Utterances**: Eliminates the need to manually create utterances for each expert

## How It Works

1. **Knowledge Organization**:
   - Knowledge is organized hierarchically in directories (groups → experts → PDF files)
   - Each group contains multiple expert directories, each expert directory contains PDF files

2. **Vector Representation**:
   - Each PDF is split into chunks and converted to embeddings
   - Each expert's knowledge is represented by a centroid vector (average of its document embeddings)
   - Each group's knowledge is represented by a weighted centroid of its experts' centroids

3. **Query Routing**:
   - When a query arrives, it's converted to an embedding vector
   - The system first finds the most similar group using cosine similarity
   - Then finds the most similar expert within that group
   - The expert's response handler is invoked to generate the final answer

## Usage

### Setup

1. Install requirements:
   ```bash
   pip install -r src/requirements.txt
   ```

2. Place PDF files in the appropriate expert directories:
   - Each PDF should contain domain-specific knowledge
   - File names must be unique within each expert directory
   - PDFs in `data/course_material/expert_for_course_1/` will be associated with that expert
   - Similarly for other experts in their respective directories

### Running Options

1. **Process PDF files and generate centroids**:
   ```bash
   python src/main.py --process
   ```
   This scans all PDF files, calculates embeddings, and updates centroid vectors for all experts and groups.

2. **Test with sample queries**:
   ```bash
   python src/main.py --test
   ```
   Tests the router with sample queries to see which expert handles each query.

3. **Query the system interactively**:
   ```bash
   python src/main.py --query 
   ```
   Enters a interactive interface where you can type queries and get responses from the routed expert.

## Future Work

1. **Integration with Vector Databases**:
   - Store document embeddings in a proper vector database like ChromaDB for scalability
   - Optimize retrieval with approximate nearest neighbor search

2. **LLM Integration**:
   - Connect expert handlers with domain-specific LLM prompts
   - Implement RAG (Retrieval Augmented Generation) for each expert

3. **Dynamic Expert Discovery**:
   - Automatically detect new expert directories and update the routing system
   - Dynamically add new experts without code changes

4. **Performance Optimization**:
   - Implement caching for frequently asked queries
   - Optimize embedding generation for large document collections

5. **Evaluation Framework**:
   - Add metrics to evaluate routing accuracy
   - Implement feedback mechanisms to improve routing over time