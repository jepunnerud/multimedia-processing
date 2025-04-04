import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Define term-document vectors for each document
document_vectors = {
    "The Dark Knight": np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "The Joker": np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    "The Matrix": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
}

# Define term-document vectors for each query
query_vectors = {
    "Query 1": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
    "Query 2": np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "Query 3": np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 0]),
}

# Define relevant documents for each query (1 for relevant, 0 for irrelevant)
relevant_documents = {
    "Query 1": ["The Matrix"],  # Relevant documents for Query 1
    "Query 2": ["The Joker"],   # Relevant documents for Query 2
    "Query 3": ["The Dark Knight", "The Joker"],  # Relevant documents for Query 3
}

# Calculate cosine similarity between each query and each document
def compute_cosine_similarity(query_vectors, document_vectors):
    results = {}
    for query_name, query_vector in query_vectors.items():
        query_results = {}
        for doc_name, doc_vector in document_vectors.items():
            # Compute cosine similarity between query and document
            similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
            query_results[doc_name] = similarity
        results[query_name] = query_results
    return results

# Compute precision and recall for a query at different cut-off points
def compute_precision_recall(query_name, similarities, relevant_docs):
    sorted_docs = sorted(similarities[query_name].items(), key=lambda x: x[1], reverse=True)  # Sort by similarity
    sorted_docs = [doc for doc, _ in sorted_docs]  # Extract document names in order of similarity
    
    precision_values = []
    recall_values = []
    retrieved = 0  # Keep track of retrieved documents
    relevant_retrieved = 0  # Keep track of relevant documents retrieved
    
    total_relevant = len(relevant_docs)
    
    # Iterate over sorted documents, progressively adding documents
    for i, doc in enumerate(sorted_docs):
        retrieved += 1
        if doc in relevant_docs:
            relevant_retrieved += 1
            
        precision = relevant_retrieved / retrieved
        recall = relevant_retrieved / total_relevant
        
        precision_values.append(precision)
        recall_values.append(recall)
        
    return precision_values, recall_values

# Get cosine similarities
similarities = compute_cosine_similarity(query_vectors, document_vectors)

# Plotting precision and recall over time for each query (separate graphs)
def plot_precision_recall_separately(similarities, relevant_documents):
    # Precision plot
    plt.figure(figsize=(10, 6))
    for query_name in similarities.keys():
        precision, _ = compute_precision_recall(query_name, similarities, relevant_documents[query_name])
        
        # Plot precision
        plt.plot(range(1, len(precision) + 1), precision, label=f'{query_name} Precision', linestyle='--')
    
    plt.xlabel('Number of Documents Retrieved')
    plt.ylabel('Precision')
    plt.title('Precision over Time')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Recall plot
    plt.figure(figsize=(10, 6))
    for query_name in similarities.keys():
        _, recall = compute_precision_recall(query_name, similarities, relevant_documents[query_name])
        
        # Plot recall
        plt.plot(range(1, len(recall) + 1), recall, label=f'{query_name} Recall', linestyle='-')
    
    plt.xlabel('Number of Documents Retrieved')
    plt.ylabel('Recall')
    plt.title('Recall over Time')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Call the plotting function
plot_precision_recall_separately(similarities, relevant_documents)
