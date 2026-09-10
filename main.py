from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from PIL import Image
from io import BytesIO

from matcher import (
    match_lost_against_found,
    match_found_against_lost
)

from database import (
    get_found_items,
    get_lost_items
)

from lost_item_service import create_lost_item
from found_item_service import create_found_item

from model import (
    get_image_embedding,
    get_text_embedding
)


app = FastAPI(
    title="Lost & Found AI",
    description="Multimodal Lost & Found matching API",
    version="1.0.0"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Lost & Found AI API is running!",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "google/siglip-base-patch16-224"
    }


# ============================================================
# CREATE LOST ITEM
# ============================================================

@app.post("/lost-items")
async def create_lost_item_endpoint(
    title: str = Form(...),
    campus: str = Form(...),
    area: str = Form(...),
    details: str = Form(""),
    category: str = Form(...),
    image_url: str = Form(...)
):
    """
    Create a persistent lost-item record.

    The image must already be uploaded to Supabase Storage.

    The service automatically generates:

        Image embedding -> SigLIP
        Text embedding  -> SigLIP

    Both embeddings are stored in Supabase.
    """

    if not image_url.strip():
        raise HTTPException(
            status_code=400,
            detail="image_url cannot be empty."
        )

    try:

        result = create_lost_item(
            title=title,
            campus=campus,
            area=area,
            details=details,
            category=category,
            image_url=image_url
        )

        return {
            "success": True,
            "message": "Lost item created successfully.",
            "lost_item": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# CREATE FOUND ITEM
# ============================================================

@app.post("/found-items")
async def create_found_item_endpoint(
    title: str = Form(...),
    campus: str = Form(...),
    area: str = Form(...),
    details: str = Form(""),
    category: str = Form(...),
    image_url: str = Form(...)
):
    """
    Create a persistent found-item record.

    The image must already be uploaded to Supabase Storage.

    The service automatically generates:

        Image embedding -> SigLIP
        Text embedding  -> SigLIP

    Both embeddings are stored in Supabase.
    """

    if not image_url.strip():
        raise HTTPException(
            status_code=400,
            detail="image_url cannot be empty."
        )

    try:

        result = create_found_item(
            title=title,
            campus=campus,
            area=area,
            details=details,
            category=category,
            image_url=image_url
        )

        return {
            "success": True,
            "message": "Found item created successfully.",
            "found_item": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# MATCH LOST ITEM AGAINST FOUND ITEMS
# ============================================================

@app.post("/match")
async def match_item(
    image: UploadFile = File(...),
    title: str = Form(...),
    campus: str = Form(...),
    area: str = Form(...),
    details: str = Form(""),
    category: str = Form(...)
):

    if (
        not image.content_type
        or not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    image_bytes = await image.read()

    try:

        lost_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # --------------------------------------------------------
    # Generate temporary embeddings for instant matching
    # --------------------------------------------------------

    print(
        "\nGenerating lost-item image embedding..."
    )

    image_embedding = get_image_embedding(
        lost_image
    )

    print(
        "Generating lost-item text embedding..."
    )

    text_embedding = get_text_embedding(
        title
    )

    lost_item = {
        "title": title,

        "location": {
            "campus": campus,
            "area": area,
            "details": details
        },

        "category": category,

        "image_embedding":
            image_embedding
            .detach()
            .cpu()
            .tolist(),

        "text_embedding":
            text_embedding
            .detach()
            .cpu()
            .tolist()
    }

    # --------------------------------------------------------
    # Get active found items
    # --------------------------------------------------------

    found_items = get_found_items()

    # --------------------------------------------------------
    # Lost -> Found
    # --------------------------------------------------------

    results = match_lost_against_found(
        lost_item=lost_item,
        found_items=found_items
    )

    matches = []

    for result in results:

        matches.append({
            "id": result["id"],
            "image": result["image_url"],
            "title": result["title"],
            "location": result["location"],
            "category": result["category"],

            "image_score":
                result["image_score"],

            "text_score":
                result["text_score"],

            "location_score":
                result["location_score"],

            "category_score":
                result["category_score"],

            "match_score":
                result["final_score"],

            "match_percentage":
                round(
                    result["final_score"] * 100,
                    2
                )
        })

    return {
        "success": True,

        "lost_item": {
            "title": title,

            "location": {
                "campus": campus,
                "area": area,
                "details": details
            },

            "category": category
        },

        "matches": matches
    }


# ============================================================
# MATCH FOUND ITEM AGAINST LOST ITEMS
# ============================================================

@app.post("/match-found")
async def match_found_item(
    image: UploadFile = File(...),
    title: str = Form(...),
    campus: str = Form(...),
    area: str = Form(...),
    details: str = Form(""),
    category: str = Form(...)
):

    if (
        not image.content_type
        or not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    image_bytes = await image.read()

    try:

        found_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # --------------------------------------------------------
    # Generate temporary embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating found-item image embedding..."
    )

    image_embedding = get_image_embedding(
        found_image
    )

    print(
        "Generating found-item text embedding..."
    )

    text_embedding = get_text_embedding(
        title
    )

    found_item = {
        "title": title,

        "location": {
            "campus": campus,
            "area": area,
            "details": details
        },

        "category": category,

        "image_embedding":
            image_embedding
            .detach()
            .cpu()
            .tolist(),

        "text_embedding":
            text_embedding
            .detach()
            .cpu()
            .tolist()
    }

    # --------------------------------------------------------
    # Get active lost items
    # --------------------------------------------------------

    lost_items = get_lost_items()

    # --------------------------------------------------------
    # Found -> Lost
    # --------------------------------------------------------

    results = match_found_against_lost(
        found_item=found_item,
        lost_items=lost_items
    )

    matches = []

    for result in results:

        matches.append({
            "id": result["id"],
            "image": result["image_url"],
            "title": result["title"],
            "location": result["location"],
            "category": result["category"],

            "image_score":
                result["image_score"],

            "text_score":
                result["text_score"],

            "location_score":
                result["location_score"],

            "category_score":
                result["category_score"],

            "match_score":
                result["final_score"],

            "match_percentage":
                round(
                    result["final_score"] * 100,
                    2
                )
        })

    return {
        "success": True,

        "found_item": {
            "title": title,

            "location": {
                "campus": campus,
                "area": area,
                "details": details
            },

            "category": category
        },

        "matches": matches
    }

