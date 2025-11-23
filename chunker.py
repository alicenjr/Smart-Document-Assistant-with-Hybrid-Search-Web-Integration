# chunker.py
def process_images_with_caption(raw_chunks, use_gemini=True):
    import base64
    import google.generativeai as genai
    from dotenv import load_dotenv
    from unstructured.documents.elements import Image, FigureCaption
    load_dotenv()

    # DEV KEY (replace with env var for prod)
    api_key = "AIzaSyCkarJ3Y8pzrK_rYMtId6xMkVUu49pLmLA"
    genai.configure(api_key=api_key)

    processed_images = []
    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Image):
            caption = None
            if idx + 1 < len(raw_chunks) and isinstance(raw_chunks[idx + 1], FigureCaption):
                caption = raw_chunks[idx + 1].text

            image_data = {
                "caption": caption if caption else "No caption",
                "image_text": chunk.text,
                "base64_image": chunk.metadata.image_base64,
                "content": chunk.text,         # fallback
                "content_type": "image/png",
                "filename": chunk.metadata.filename,
            }

            if use_gemini:
                model = genai.GenerativeModel("gemini-2.5-flash")
                image_binary = base64.b64decode(image_data["base64_image"])
                prompt = (
                    f"Describe the image in detail. "
                    f"Caption: '{image_data['caption']}'. "
                    f"Extracted text: '{image_data['image_text']}'."
                )
                resp = model.generate_content([
                    prompt,
                    {"mime_type": "image/png", "data": image_binary},
                ])
                image_data["content"] = resp.text

            processed_images.append(image_data)
    return processed_images

def process_tables_with_description(raw_chunks, use_gemini=True):
    import google.generativeai as genai
    from dotenv import load_dotenv
    from unstructured.documents.elements import Table
    load_dotenv()

    api_key = "AIzaSyCkarJ3Y8pzrK_rYMtId6xMkVUu49pLmLA"
    if not api_key:
        raise ValueError("Gemini API key not set.")
    genai.configure(api_key=api_key)

    processed_tables = []
    for element in raw_chunks:
        if isinstance(element, Table):
            table_data = {
                "table_as_html": element.metadata.text_as_html,
                "table_text": element.text,
                "content": element.text,
                "content_type": "table",
                "filename": element.metadata.filename,
            }
            if use_gemini:
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = (
                    "Analyze the following HTML table and provide a detailed, direct description of "
                    "its structure, key values, and notable patterns. "
                    f"HTML: {table_data['table_as_html']}"
                )
                resp = model.generate_content([prompt])
                table_data["content"] = resp.text
            processed_tables.append(table_data)
    return processed_tables

def create_semantic_chunks(text_chunks):
    from unstructured.documents.elements import CompositeElement
    processed_chunks = []
    for chunk in text_chunks:
        if isinstance(chunk, CompositeElement):
            processed_chunks.append({
                "content": chunk.text,
                "content_type": "text",
                "filename": chunk.metadata.filename if chunk.metadata else None
            })
    return processed_chunks

if __name__ == "__main__":
    from unstructured.partition.pdf import partition_pdf
    text_chunks = partition_pdf(
        filename="C:\\Users\\leoli\\Documents\\langchain_tools\\hunter.pdf",
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=2000,
        combine_text_under_n_chars=500,
        new_after_n_chars=1500
    )
    print(text_chunks)
    semantics_chunks = create_semantic_chunks(text_chunks)

