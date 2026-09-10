import torch
from transformers import AutoProcessor, SiglipModel


MODEL_NAME = "google/siglip-base-patch16-224"


print("Loading SigLIP model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SiglipModel.from_pretrained(MODEL_NAME)

model.eval()

print("SigLIP model loaded successfully!")


def get_image_embedding(image):
    """
    Convert an image into a normalized SigLIP embedding.
    """

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.get_image_features(**inputs)

        if hasattr(output, "pooler_output"):
            embedding = output.pooler_output
        else:
            embedding = output

    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    return embedding.squeeze(0)


def get_text_embedding(text):
    """
    Convert text into a normalized SigLIP embedding.
    """

    inputs = processor(
        text=[text],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.get_text_features(**inputs)

        if hasattr(output, "pooler_output"):
            embedding = output.pooler_output
        else:
            embedding = output

    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    return embedding.squeeze(0)
