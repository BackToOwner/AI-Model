import requests

from PIL import Image
from io import BytesIO

from database import (
    supabase,
    get_lost_items
)

from model import (
    get_image_embedding,
    get_text_embedding
)

from matcher import match_found_against_lost

def create_found_item(
    title,
    campus,
    area,
    details,
    category,
    image_url
):
    """
    Create a persistent found-item record and automatically
    generate both SigLIP image and text embeddings.
    """

    print("\nCreating found item...")

    # --------------------------------------------------------
    # 1. Create database record
    # --------------------------------------------------------

    response = (
        supabase
        .table("found_items")
        .insert({
            "title": title,
            "campus": campus,
            "area": area,
            "details": details,
            "category": category,
            "image_url": image_url
        })
        .execute()
    )

    if not response.data:
        raise Exception(
            "Failed to create found item."
        )

    item = response.data[0]
    item_id = item["id"]

    print(
        f"Found item created: {item_id}"
    )

    # --------------------------------------------------------
    # 2. Download image
    # --------------------------------------------------------

    print(
        "Downloading found-item image..."
    )

    image_response = requests.get(
        image_url,
        timeout=30
    )

    image_response.raise_for_status()

    image = Image.open(
        BytesIO(
            image_response.content
        )
    ).convert("RGB")

    print(
        "Image downloaded:",
        image.size
    )

    # --------------------------------------------------------
    # 3. Generate image embedding
    # --------------------------------------------------------

    print(
        "Generating image embedding..."
    )

    image_embedding = (
        get_image_embedding(image)
    )

    image_embedding_list = (
        image_embedding
        .detach()
        .cpu()
        .tolist()
    )

    print(
        "Image embedding dimension:",
        len(image_embedding_list)
    )

    # --------------------------------------------------------
    # 4. Generate text embedding
    # --------------------------------------------------------

    print(
        "Generating text embedding..."
    )

    text_embedding = (
        get_text_embedding(title)
    )

    text_embedding_list = (
        text_embedding
        .detach()
        .cpu()
        .tolist()
    )

    print(
        "Text embedding dimension:",
        len(text_embedding_list)
    )

    # --------------------------------------------------------
    # 5. Save both embeddings
    # --------------------------------------------------------

    print(
        "Saving embeddings to Supabase..."
    )

    update_response = (
        supabase
        .table("found_items")
        .update({
            "image_embedding":
                image_embedding_list,

            "text_embedding":
                text_embedding_list
        })
        .eq("id", item_id)
        .execute()
    )

    if not update_response.data:
        raise Exception(
            "Failed to save embeddings."
        )

    print(
        "Both embeddings saved successfully."
    )

    # --------------------------------------------------------
    # 6. Automatically search for matching lost items
    # --------------------------------------------------------

    print(
        "\nSearching for matching lost items..."
    )

    found_item = {
        "id": item_id,
        "title": title,
        "location": {
            "campus": campus,
            "area": area,
            "details": details
        },
        "category": category,
        "image_embedding": image_embedding_list,
        "text_embedding": text_embedding_list
    }

    lost_items = get_lost_items()

    matches = match_found_against_lost(
        found_item,
        lost_items
    )

    print(
        f"Found {len(matches)} possible matches."
    )

    # --------------------------------------------------------
    # 6. Return result
    # --------------------------------------------------------

    return {
        "id": item_id,
        "title": title,
        "campus": campus,
        "area": area,
        "details": details,
        "category": category,
        "image_url": image_url,

        "image_embedding_dimension":
            len(image_embedding_list),

        "text_embedding_dimension":
            len(text_embedding_list),

        "matches": matches
    } 
