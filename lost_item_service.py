import requests

from PIL import Image
from io import BytesIO

from database import (
    supabase,
    get_found_items
)

from model import (
    get_image_embedding,
    get_text_embedding
)

from matcher import match_lost_against_found

def create_lost_item(
    title,
    campus,
    area,
    details,
    category,
    image_url
):
    """
    Create a lost-item record and automatically
    generate both SigLIP image and text embeddings.
    """

    # --------------------------------------------------
    # 1. INSERT LOST ITEM
    # --------------------------------------------------

    print("\nCreating lost item...")

    response = (
        supabase
        .table("lost_items")
        .insert({
            "title": title,
            "campus": campus,
            "area": area,
            "details": details,
            "category": category,
            "image_url": image_url,
            "status": "active"
        })
        .execute()
    )

    if not response.data:
        raise Exception(
            "Failed to create lost item."
        )

    item = response.data[0]

    item_id = item["id"]

    print(
        f"Lost item created: {item_id}"
    )


    # --------------------------------------------------
    # 2. DOWNLOAD IMAGE
    # --------------------------------------------------

    print(
        "Downloading lost-item image..."
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


    # --------------------------------------------------
    # 3. IMAGE EMBEDDING
    # --------------------------------------------------

    print(
        "Generating image embedding..."
    )

    image_embedding = (
        get_image_embedding(
            image
        )
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


    # --------------------------------------------------
    # 4. TEXT EMBEDDING
    # --------------------------------------------------

    print(
        "Generating text embedding..."
    )

    text_embedding = (
        get_text_embedding(
            title
        )
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


    # --------------------------------------------------
    # 5. SAVE BOTH EMBEDDINGS
    # --------------------------------------------------

    print(
        "Saving embeddings to Supabase..."
    )

    update_response = (
        supabase
        .table("lost_items")
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

    
    print("\nSearching for matching found items...")

    lost_item = {
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

    found_items = get_found_items()

    matches = match_lost_against_found(
        lost_item,
        found_items
    )

    print(f"Found {len(matches)} possible matches.")

    # --------------------------------------------------
    # 6. RETURN RESULT
    # --------------------------------------------------

    return {
        "id": item_id,
        "title": title,
        "campus": campus,
        "area": area,
        "details": details,
        "category": category,
        "image_url": image_url,
        "status": "active",
        "image_embedding_dimension": len(image_embedding_list),
        "text_embedding_dimension": len(text_embedding_list),
        "matches": matches
    }