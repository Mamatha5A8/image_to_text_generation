import streamlit as st
from PIL import Image
from googletrans import Translator
from diffusers import StableDiffusionPipeline
import torch

# Initialize translator and StableDiffusionPipeline
translator = Translator()
image_gen_model = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token='your_hugging_face_auth_token',
    guidance_scale=9
)

# Function to generate image based on text prompt
def generate_image(prompt, model):
    image = model(
        prompt,
        num_inference_steps=35,
        generator=torch.Generator("cuda").manual_seed(42),
        guidance_scale=9
    ).images[0]
    image = image.resize((900, 900))
    return image

# Function to translate text
def get_translation(text, dest_lang):
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text

# Streamlit app
def main():
    st.title("Image Generation")
    prompt = st.text_input("Enter text prompt:")
    if st.button("Generate Image"):
        # Translate the prompt to English
        translated_prompt = get_translation(prompt, "en")
        st.write("Translated Prompt:", translated_prompt)
        
        # Generate image based on the translated prompt
        image = generate_image(translated_prompt, image_gen_model)
        st.image(image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()

