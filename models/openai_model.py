import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import openai

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAIWrapper:
    def __init__(self, model_id="gpt-4o-mini"):
        """
        Wrapper for OpenAI's GPT-4o (or similar multimodal models).
        Args:
            model_id (str): Which OpenAI model to use, e.g., "gpt-4o-mini"
        """
        self.model_id = model_id

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for OpenAI API."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate(self, image, instruction: str) -> str:
        if not isinstance(image, Image.Image):
            image = image.convert("RGB")

        image_b64 = self._image_to_base64(image)

        response = openai.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ],
                }
            ],
            max_tokens=150,
        )

        return response.choices[0].message.content.strip()
