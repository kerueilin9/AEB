from unstructured.partition.pdf import partition_pdf

output_path = "./images/test"

# Get elements
raw_pdf_elements = partition_pdf(
    filename="./docs/SElab_Industry_Academia_Collaboration.pdf",
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    extract_image_block_output_dir=output_path,
)