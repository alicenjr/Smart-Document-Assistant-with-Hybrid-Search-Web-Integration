# ingestion.py
from helper import get_opensearch_client, get_embedding

def create_index_if_not_exists(client, index_name: str):
    """
    Create a k-NN index suitable for 768-d embeddings.
    Uses OpenSearch knn_vector mapping (not Elasticsearch dense_vector).
    """
    if client.indices.exists(index=index_name):
        print(f"Index {index_name} already exists. Deleting and recreating...")
        client.indices.delete(index=index_name)

    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "content_type": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {
                        "name": "hnsw",
                        "engine": "nmslib",
                        "space_type": "cosinesimil",
                        "parameters": {"m": 16, "ef_construction": 128}
                    }
                }
            }
        }
    }

    try:
        client.indices.create(index=index_name, body=body)
        print(f"Index {index_name} created successfully.")
    except Exception as e:
        print(f"Error creating index {index_name}: {e}")
        raise

def prepare_chunks_for_ingestion(chunks):
    """
    Takes list of dicts having keys: content, content_type, filename.
    Adds 'embedding' using Ollama embeddings (768-d).
    """
    prepared = []
    for idx, chunk in enumerate(chunks):
        content = (chunk or {}).get("content", "")
        if not content or not content.strip():
            print(f"Skipping empty chunk at index {idx}")
            continue

        try:
            emb = get_embedding(content)  # 768-d from nomic-embed-text
        except Exception as e:
            print(f"Embedding failed at index {idx}: {e}")
            continue

        prepared.append({
            "content": content,
            "content_type": chunk.get("content_type", "text"),
            "filename": chunk.get("filename"),
            "embedding": emb
        })
    return prepared

def ingest_chunks_into_opensearch(client, index_name, chunks):
    """Bulk-ingest prepared chunks into OpenSearch."""
    from opensearchpy import helpers
    if not chunks:
        print("No chunks to ingest.")
        return
    actions = ({"_index": index_name, "_source": doc} for doc in chunks)
    try:
        helpers.bulk(client, actions, request_timeout=120)
        print(f"Ingested {len(chunks)} chunks into index {index_name}.")
    except Exception as e:
        print(f"Error ingesting chunks into index {index_name}: {e}")
        raise

def ingest_all_content_into_opensearch(processed_images, processed_tables, semantic_chunks, index_name):
    
    client = get_opensearch_client("localhost", 9200)
    create_index_if_not_exists(client, index_name)

    image_chunks = prepare_chunks_for_ingestion(processed_images)
    ingest_chunks_into_opensearch(client, index_name, image_chunks)

    table_chunks = prepare_chunks_for_ingestion(processed_tables)
    ingest_chunks_into_opensearch(client, index_name, table_chunks)

    text_chunks = prepare_chunks_for_ingestion(semantic_chunks)
    ingest_chunks_into_opensearch(client, index_name, text_chunks)

if __name__ == "__main__":
    from unstructured.partition.pdf import partition_pdf
    from chunker import (
        process_images_with_caption,
        process_tables_with_description,
        create_semantic_chunks,
    )

    pdf_path = "C:\\Users\\leoli\\Documents\\langchain_tools - Copy\\niraj.pdf"

    # Extract images & tables (hi_res may warn about size params; harmless for our flow)
    raw_chunks_media = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy=None,
    )
    processed_images = process_images_with_caption(raw_chunks_media, use_gemini=True)
    processed_tables = process_tables_with_description(raw_chunks_media, use_gemini=True)

    # Extract semantic text chunks
    raw_chunks_text = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=2000,
        combine_text_under_n_chars=500,
        new_after_n_chars=1500,
    )
    semantics_chunks = create_semantic_chunks(raw_chunks_text)

    index_name = "pdf_content_index"
    ingest_all_content_into_opensearch(
        processed_images,
        processed_tables,
        semantics_chunks,
        index_name
    )
    print("Ingestion complete.")