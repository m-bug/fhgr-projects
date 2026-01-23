###
# This project uses the OpenAI Images API.
# An API key is required and must be provided via the
# OPENAI_API_KEY environment variable.
# Before executing this script: Create a .env file and fill in the varianle OPENAI_API_KEY=XY
###

###
# Getting started:
# pip install openai pillow matplotlib python-dotenv
###


import os
import base64
import time
from io import BytesIO

import matplotlib.pyplot as plt
import PIL.Image
from dotenv import load_dotenv
from openai import OpenAI


# =====================================================
# 1. Environment & Client
# =====================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please create a .env file.")

client = OpenAI(api_key=api_key)
IMAGE_MODEL = "gpt-4.1"

# =====================================================
# 2. Prompt-Gruppen
# =====================================================
prompt_groups = {
    "Control: Style": [
        "A person sitting at a desk, working on a laptop, photorealistic, natural lighting.",
        "A person sitting at a desk, working on a laptop, watercolor illustration, soft colors.",
        "A person sitting at a desk, working on a laptop, pixel art style, 16-bit retro aesthetic."
    ],
    "Randomness": [
        "A person sitting at a desk, working on a laptop, photorealistic.",
        "A person sitting at a desk, working on a laptop, photorealistic.",
        "A person sitting at a desk, working on a laptop, photorealistic."
    ],
    "Limits": [
        "A person sitting at a desk, left hand clearly visible with exactly six fingers.",
        "A blue sphere with a red cube inside it, physically consistent and realistic.",
        "A clock showing exactly 10:07, with both hour and minute hands clearly visible."
    ]
}

# =====================================================
# 3. Bildgenerierung
# =====================================================
def generate_image(prompt: str) -> PIL.Image.Image:
    response = client.responses.create(
        model=IMAGE_MODEL,
        input=prompt,
        tools=[{"type": "image_generation"}],
    )

    # Extrahiere das Base64 Bild
    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]
    if not image_data:
        raise RuntimeError(f"No image returned for prompt: {prompt}")

    image_base64 = image_data[0]
    image_bytes = base64.b64decode(image_base64)
    return PIL.Image.open(BytesIO(image_bytes))


def generate_image_with_retry(prompt: str, retries=3, delay=5) -> PIL.Image.Image:
    """
    Retry mechanism for image generation.
    """
    for attempt in range(retries):
        try:
            return generate_image(prompt)
        except RuntimeError as e:
            print(f"Warning: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"Failed to generate image after {retries} retries: {prompt}")

# =====================================================
# 4. Sanity-Test (Minimal Prompt)
# =====================================================
try:
    test_prompt = "A cute cat, photorealistic, 1024x1024"
    print("Generating test image to check API connectivity...")
    test_img = generate_image_with_retry(test_prompt, retries=2, delay=5)
    print("Test image generated successfully!")
except Exception as e:
    print(f"Error generating test image: {e}")
    raise SystemExit("Aborting: Check API key, network or prompt validity.")

# =====================================================
# 5. Alle Gruppen-Bilder erzeugen
# =====================================================
generated_images = {}
for group_name, prompts in prompt_groups.items():
    print(f"Generating group: {group_name}")
    generated_images[group_name] = []

    for i, prompt in enumerate(prompts):
        print(f"  → Prompt {i + 1}")
        img = generate_image_with_retry(prompt)
        generated_images[group_name].append(img)

# =====================================================
# 6. Raster mit Matplotlib
# =====================================================
rows = len(prompt_groups)
cols = max(len(p) for p in prompt_groups.values())

fig, axes = plt.subplots(
    rows, cols, figsize=(cols * 4, rows * 4)
)
if rows == 1:
    axes = [axes]

for row_idx, (group, images) in enumerate(generated_images.items()):
    for col_idx in range(cols):
        ax = axes[row_idx][col_idx]
        if col_idx < len(images):
            ax.imshow(images[col_idx])
            if col_idx == 0:
                ax.set_ylabel(group, fontsize=11, rotation=90, labelpad=10)
        ax.axis("off")

##plt.suptitle("From Text to Image: Control, Randomness and Limits", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.90)

output_file = "m04_text_to_image_grid_v4.png"
plt.savefig(output_file, dpi=250)
plt.show()

print(f"\nSaved result as: {output_file}")
