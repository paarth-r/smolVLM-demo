import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import cv2
import os
import json
import re


class VLMAnalyzer:
    def __init__(self):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pass
    def capture_image(self, name="image.jpg"):
        """
        Captures an image from the webcam and saves it to the 'images' directory. (image.jpg by default)
        Will do not replace existing image if image.jpg already exists.
        Press 's' to save image when camera is rolling
        """
        if not os.path.exists("images/" + name + ".*"):
            cap = cv2.VideoCapture(0)
            print("Press 's' to save image")
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
                    cv2.imwrite("images/" + name + ".*", frame)
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Image already exists")
    def analyze_image(self, imagepath="images/image.jpg"):
        """
        Analyzes the image(images/image.jpg by default) and returns a JSON object with the following fields:
        - description: a textual description of the main object in the image
        - size: the estimated size of the main object in inches
        - properties: the properties of the main object, including 'color', 'texture', 'rigidity' and 'grippiness'
        """
        # Load images
        image = load_image(imagepath)

        # Initialize processor and model
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if self.DEVICE == "cuda" else "eager",
        ).to(self.DEVICE)

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
        inputs = inputs.to(self.DEVICE)

        # Generate outputs
        generated_ids = model.generate(**inputs, max_new_tokens=500)  # Increased token limit for JSON
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        # Try to extract and parse JSON from the response

        try:
            # Try to find JSON in the response (in case there's any extra text)
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Parse and pretty print the JSON
                parsed_json = json.loads(json_str)
                return json.dumps(parsed_json, indent=2)
            else:
                return f"no json found in output\nraw output: {generated_text}"
        except json.JSONDecodeError as e:
            return f"json decode error: {e}\nraw output: {generated_text}"
        