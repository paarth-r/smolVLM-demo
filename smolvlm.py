import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import cv2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

if not os.path.exists("images/image.jpg"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            cv2.imwrite("images/image.jpg", frame)
            break
    cap.release()
    cv2.destroyAllWindows()

# Load images
image = load_image("images/image.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages with structured output request
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Analyze this image and provide a structured response in JSON format with the following fields: 'description' (a textual description of the main object in the image), 'size' (the estimated size of the main object in inches), and 'properties' (the properties of the main object, including 'color', 'texture', 'rigidity' and 'slipperiness'). Only respond with valid JSON, no additional text. 'size', 'properties', and 'description' are mandatory."}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)  # Increased token limit for JSON
generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

# Try to extract and parse JSON from the response
import json
import re

try:
    # Try to find JSON in the response (in case there's any extra text)
    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # Parse and pretty print the JSON
        parsed_json = json.loads(json_str)
        print(json.dumps(parsed_json, indent=2))
    else:
        print("Could not parse JSON from response. Raw output:")
        print(generated_text)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print("Raw model output:")
    print(generated_text)
