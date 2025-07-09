# SmolVLM Image Analyzer

This Python script captures an image using your webcam and analyzes it using the [SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) model from Hugging Face. The output is a structured JSON object describing the main object in the image.

## Features

- Captures an image from webcam if not already saved
- Uses `SmolVLM-Instruct` Vision-Language model for analysis
- Returns:
  - `description` of the main object
  - Estimated `size` in inches
  - `properties` such as color, texture, rigidity, and slipperiness
- Outputs only valid, parseable JSON
- Given image (airpods) produces following description:
```
{
  "description": "A pair of white AirPods in their charging case",
  "size": "1.5 x 1.5 x 0.75 inches",
  "properties": {
    "color": "white",
    "texture": "smooth",
    "rigidity": "very rigid",
    "slipperiness": "very slippery"
  }
}
```
The output needs some fine tuning but will work for rudimentary applications.

## Requirements

- Python 3.8+
- PyTorch (with CUDA or MPS support if available)
- OpenCV
- Transformers (🤗 Hugging Face)
- PIL (Pillow)

Install dependencies:

```bash
pip install torch torchvision transformers accelerate opencv-python pillow}

