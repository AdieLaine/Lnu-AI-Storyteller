# System-related
import os
import sys
import tempfile
import time

# File and IO operations
import json
import jsonlines
import soundfile as sf

# Data processing and manipulation
import numpy as np
import pandas as pd
import random

# Image processing
from PIL import Image

# Network requests
import requests

# Text-to-speech
from gtts import gTTS
from pydub import AudioSegment

# Type hints
from typing import Dict, List, Tuple, Optional

# Language model
import openai

# Web application framework
import streamlit as st
from streamlit_option_menu import option_menu


# Environment variables
from dotenv import load_dotenv

CUSTOM_CSS = """
    <style>
        .big-font {
            font-size: 20px;
            font-weight: bold;
        }
        .red-text {
            color: crimson;
        }
        .green-text {
            color: #42f5ad;
        }
        .blue-text {
            color: #4287f5;
        }
        .selected-word {
            font-weight: bold;
            color: #f542f7;
        }
        .yellow-text {
            color: #FFD700;
        }
        .custom-option-menu select {
            font-weight: bold;
            color: #FF6347;
            background-color: #FFF;
        }
    </style>
"""

# Streamlit configurations
st.set_page_config(
    page_title="Lnu-AI - An Indigenous AI System",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/AdieLaine/lnu-ai',
        'Report a bug': 'https://github.com/AdieLaine/lnu-ai/issues',
        'About': """
            # Lnu-AI
            Welcome to Lnu-AI! This application is dedicated to helping people learn and appreciate the Mi'kmaq language, an indigenous language of Eastern Canada and the United States. 

            ## About Mi'kmaq Language
            The Mi'kmaq language is a rich, polysynthetic language with a deep historical and cultural significance. It is, however, at risk of being lost as the number of fluent speakers decreases.

            ## The Lnu-AI Project
            Lnu-AI utilizes advanced AI technologies to provide a platform for learning, using, and preserving the Mi'kmaq language. It offers various features like chat functionality, storytelling, and deep linguistic analysis to facilitate language learning and appreciation.

            ## Your Contribution
            As an open-source project, we welcome contributions that help improve Lnu-AI and further its mission to preserve the Mi'kmaq language. Please visit our [GitHub](https://github.com/AdieLaine/lnu-ai) page for more information.
            
            Enjoy your journey with the Mi'kmaq language!
        """
    }
)


@st.cache_data
def load_all_word_details(file):
    """
    Load word details from a JSON file.

    Parameters:
    file (str): Path to the JSON file containing the word details.

    Returns:
    all_word_details (dict): Dictionary of word details loaded from the JSON file.
    """
    full_path = os.path.join("data", file)
    if os.path.isfile(full_path):
        with open(full_path, 'r') as f:
            all_word_details = json.load(f)
        return all_word_details
    else:
        return None
    
file = 'all_word_details.json'

all_word_details: dict = load_all_word_details(file)


@st.cache_data
def load_theme_and_story(jsonl_file):
    """
    Load all themes from a JSONL file.

    Args:
        jsonl_file (str): Path to the JSONL file containing the themes.
        
    Returns:
        themes (list): A list containing all themes, or None if an error occurred.
    """
    try:
        full_path = os.path.join("data", jsonl_file)
        with jsonlines.open(full_path) as reader:
            themes = list(reader)  # Read all themes into a list
        return themes
    except (FileNotFoundError, jsonlines.jsonlines.InvalidLineError) as e:
        st.error(f"Error in loading themes: {str(e)}")
        return None

jsonl_file = "mikmaq_semantic.jsonl"

themes = load_theme_and_story(jsonl_file)


def get_env_variable(name, default=None):
    """
    Returns the value of the environment variable with the given name.
    First tries to fetch it from Streamlit secrets, and if not available,
    falls back to the local environment. If it's not found in either place,
    returns the default value if provided.
    
    Args:
        name (str): The name of the environment variable.
        default (str, optional): The default value to be returned in case the environment variable is not found.

    Returns:
        str: The value of the environment variable, or the default value.
    """
    if st.secrets is not None and name in st.secrets:
        # Fetch the secret from Streamlit secrets
        return st.secrets[name]
    else:
        # Try to get the secret from the local environment
        value = os.getenv(name)

        if value is None and default is not None:
            # If the environment variable is not found and a default value is provided,
            # return the default value
            return default
        else:
            return value


@st.cache_resource(show_spinner=False)
def load_env_variables_and_data():
    """
    Loads Lnu-AI Assistant environment variables and data.

    Returns:
    dict: A dictionary containing the loaded environment variables and data.
    """
    api_keys, tts_settings, local_data_files, models = load_env_variables()
    all_word_details = load_all_word_details(local_data_files.get("all_word_details"))


    return {
        "api_keys": api_keys,
        "tts_settings": tts_settings,
        "local_data_files": local_data_files,
        "models": models,
        "all_word_details": all_word_details,
        "CUSTOM_CSS": CUSTOM_CSS
    }


@st.cache_resource(show_spinner=False)
def load_env_variables():
    """
    Load all the environment variables required for the Lnu-AI Assistant.

    Returns:
    Tuple: A tuple containing the loaded environment variables.
    """
    api_keys = {
        "openai": get_env_variable("OPENAI_API_KEY"),
    }

    tts_settings = {
        "tts_audio": get_env_variable('TTS_AUDIO'),
        "local_tts": get_env_variable('LOCAL_TTS'),
    }

    local_data_files = {
        "all_word_details": (get_env_variable("ALL_WORD_DETAILS") + '.json') if get_env_variable("ALL_WORD_DETAILS") else None,
    }

    models = {
        "chat_model": get_env_variable("CHAT_MODEL_SELECTION", default="gpt-4-0613"),
        "fine_tuned_model_dictionary": get_env_variable("FINE_TUNED_MODEL_DICTIONARY"),
        "fine_tuned_model_data": get_env_variable("FINE_TUNED_MODEL_DATA"),
    }

    openai.api_key = api_keys["openai"]

    return api_keys, tts_settings, local_data_files, models

def generate_audio(text, tts_service):
    """
    Generate audio from text using a TTS service.

    Parameters:
    text (str): The input text.
    tts_service (str): The TTS service to use (e.g., "gtts", "st.audio").

    Returns:
    str: Path to the generated audio file.
    """
    tts_service = tts_service.lower()  # convert to lower case for case-insensitive comparison
    if tts_service == 'gtts':
        tts = gTTS(text=text, lang='en')  # Replace 'en' with the appropriate language if needed

        # Save the speech audio into a temporary mp3 file
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_mp3.name)  

        # Convert the temporary mp3 file to wav format
        audio = AudioSegment.from_mp3(temp_mp3.name)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format='wav')

        # Return the path to the generated wav file
        return temp_wav.name
    elif tts_service == 'st.audio':
        return text
    else:
        raise ValueError(f"Invalid Text-to-Speech service: {tts_service}")


def generate_openai_images(prompt, role="DALL-E", context="In the creative and vibrant style of Norval Morrisseau, using colorful Mi'kmaq themes"):
    """
    Generates an image using the OpenAI's DALL-E model.

    Args:
        prompt (str): The main role for the image generation.
        context (str, optional): Context to provide to the image generation.
                                 Defaults to "In the creative and vibrant style of Norval Morrisseau, using colorful Mi'kmaq themes".

    Returns:
        str: URL of the generated image if successful, else None.
    """
    try:
        full_prompt = f"{context} {prompt}"
        truncated_prompt = full_prompt[:300]
        prompt_settings = {
            "model": "image-alpha-001",
            "prompt": truncated_prompt,
        }
        response_settings = {
            "num_images": 1,
            "size": "1024x1024",
            "response_format": "url"
        }
        openai_key = os.getenv("OPENAI_API_KEY", "your_default_key")  # Use your default key here
        if not openai_key:
            st.info("Environment variable for OPENAI_API_KEY is not set, using default key: your_default_key")
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_key}"
            },
            json={
                **prompt_settings,
                **response_settings
            }
        )
        response.raise_for_status()  # Raise an exception for any HTTP errors
        image_url = response.json()["data"][0]["url"]
        return image_url
    except (requests.RequestException, ValueError) as e:
        st.error(f"Error in generating images: {str(e)}")
        return None


@st.cache_resource(show_spinner=False)
def enhance_with_gpt(prompt, final_reply, models):
    """
    Enhances the reply with GPT model by sending the conversation for completion.

    Args:
        prompt (str): User's message.
        final_reply (str): Assistant's reply.
        models (dict): Dictionary of trained models.

    Returns:
        str: Enhanced reply.
    """
    model_name = models.get("chat_model", "gpt-4-0613")
    try:
        gpt_messages = [
            {"role": "system", "content": "You are Lnu-AI, an AI developed to promote and preserve the Mi'kmaq language and culture."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_reply}
        ]
    
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=gpt_messages,
            max_tokens=2500,  
            temperature=0.5, 
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )

        # Extract reply from response
        if 'choices' in response and len(response['choices']) > 0:
            if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                gpt_reply = response['choices'][0]['message']['content']
                return gpt_reply

    except Exception as e:
        print(f"Error with {model_name}: {e}")

    return final_reply


@st.cache_data
def get_tts_service(tts_settings: dict) -> str:
    """
    Function to determine the appropriate Text-to-Speech service based on the settings.

    Args:
        tts_settings (dict): Text-to-Speech settings.

    Returns:
        str: Name of the Text-to-Speech service.
    """
    tts_service = next((service for service, flag in tts_settings.items() if flag.lower() == 'yes'), 'gtts')
    return tts_service



def get_theme_and_story_word(themes: Optional[List[Dict[str, str]]], selected_word: str, meaning: str) -> Tuple[str, str, str]:
    """
    Function to handle theme and story word selection.
    """
    if not themes:
        raise ValueError("No themes provided.")

    selected_theme = select_random_theme(themes)
    return replace_placeholders_in_theme(selected_theme, selected_word, meaning)



def select_random_theme(themes: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Selects a random theme from the provided list.

    Args:
        themes (List[Dict[str, str]]): The list of themes to choose from.

    Returns:
        Dict[str, str]: The selected theme.
    """
    return random.choice(themes)



def replace_placeholders_in_theme(theme: Dict[str, str], word: str, meaning: str) -> Tuple[str, str, str]:
    """
    Replaces placeholders in a theme with the provided word and meaning.

    Args:
        theme (Dict[str, str]): The theme.
        word (str): The word to replace the '{word}' placeholder.
        meaning (str): The meaning to replace the '{meaning}' placeholder.

    Returns:
        Tuple[str, str, str]: A tuple containing the theme, story word, and image theme.

    Raises:
        KeyError: If a required key is missing from the theme.
    """
    try:
        theme_text = theme['theme'].replace('{word}', word).replace('{meaning}', meaning)
        story_word = theme['story_word']
        image_theme = theme['image_theme'].replace('{word}', word).replace('{meaning}', meaning)
    except KeyError as e:
        raise KeyError(f"Required key missing from theme: {str(e)}")

    return theme_text, story_word, image_theme



def generate_story(all_word_details: dict, theme: str, story_word: str, image_theme: str) -> Tuple[str, str, str]:
    """
    Function to generate a story using OpenAI's GPT-4 model. Interacts with the OpenAI API to create a conversation
    and uses the returned message content as the generated story.

    Args:
        all_word_details (dict): Dictionary of all word details.
        theme (str): The theme for the story.
        story_word (str): The main word for the story.
        image_theme (str): The theme for the image.

    Returns:
        str: The generated story text.
        str: The main word for the story.
        str: The theme for the image.
    """

    # The model used for generating the story is retrieved from environment variables.
    # If there is no such variable defined, use "gpt-4-0613" as the default model.
    # You may want to replace this with a different model name if a new version is available.
    

    # Define the system's role and content. The system plays the role of a Mi'kmaq storyteller.
    prompt_system = {
        'role': 'system',
        'content': "You are a Mi'kmaq storyteller, or an 'a'tugwewinu', skilled in weaving captivating tales about the Mi'kmaq people and their rich cultural heritage."
    }

    # Define the initial part of the story as an 'Assistant' message.
    initial_story = {
        'role': 'assistant',
        'content': f"Let us embark on a journey with a theme of '{theme}', centered around the word '{story_word}'."
    }

    # Create a conversation with OpenAI's Chat models.
    # Parameters like max_tokens, temperature, top_p, frequency_penalty, and presence_penalty can be tweaked
    # to adjust the output from the model.
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[prompt_system, initial_story],
        max_tokens=2000,  # Maximum length of the generated text. Consider adjusting this for longer/shorter outputs.
        temperature=0.5,  # Controls the randomness of the output. Higher values (closer to 1) make output more random.
        top_p=1.0,  # Controls the nucleus sampling. Lower values can make the output more focused.
        frequency_penalty=0.5,  # Controls penalizing new tokens based on their frequency.
        presence_penalty=0.5,  # Controls penalizing new tokens based on their presence.
    )

    # This is the generated story text.
    story_text = response['choices'][0]['message']['content']

    # Return the generated story, story word, and image theme.
    return story_text, story_word, image_theme


def process_story_generation(selected_word: str, all_word_details: dict, generate_images: bool, tts_settings: dict) -> None:
    """
    Function to generate and display a story, its audio, and images.

    Args:
        selected_word (str): The selected word for the story.
        all_word_details (dict): Dictionary of all word details.
        generate_images (bool): Flag to decide if images need to be generated.
        tts_settings (dict): Text-to-Speech settings.
    """
    if not selected_word:
        st.info("Please enter a word for the story")
        return

    with st.spinner("Generating story..."):
        try:
            word_detail = all_word_details[selected_word]
            meaning = word_detail.get('meanings', [])[0]  # get the first meaning
            themes = load_theme_and_story("mikmaq_semantic.jsonl")
            theme, story_word, image_theme = get_theme_and_story_word(themes, selected_word, meaning)
            story_text, _, _ = generate_story(selected_word, theme, story_word, image_theme)

            tts_service = get_tts_service(tts_settings)
            audio = generate_audio(story_text, tts_service)
            display_story_and_audio(story_text, audio)

            st.success("Story generation completed!")
        except Exception as e:
            st.error(f"Error in generating story or audio: {str(e)}")

    if generate_images:
        generate_and_display_images(story_text, image_theme)


def display_story_and_audio(story_text: str, audio: Optional[str]) -> None:
    """
    Function to display the story text and its audio.

    Args:
        story_text (str): The generated story text.
        audio (Optional[str]): Path to the audio file.
    """
    story_container = st.container()
    with story_container:
        st.markdown(
            f"<div style='background-color: #2f4f4f; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);'>"
            f"<p style='font-size: 18px; font-weight: bold; margin-bottom: 10px;'>Story text:</p>"
            f"<p style='font-size: 16px; line-height: 1.5;'>{story_text}</p>"
            "</div>",
            unsafe_allow_html=True
        )
        if audio is not None and os.path.isfile(audio):
            with open(audio, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            os.remove(audio)  # Delete the temporary audio file after playing


def generate_and_display_images(story_text: str, image_theme: str) -> None:
    """
    Function to generate and display images for different parts of the story.

    Args:
        story_text (str): The generated story text.
        image_theme (str): The theme for the image.
    """
    story_parts = [
        story_text[:len(story_text) // 3],
        story_text[len(story_text) // 3: 2 * len(story_text) // 3],
        story_text[2 * len(story_text) // 3:],
    ]

    image_container = st.container()
    for i, part in enumerate(story_parts):
        image_url = generate_openai_images(part + ' ' + image_theme)
        if image_url:
            image_container.image(image_url, width=None, clamp=True, channels="RGB", output_format="png") # image_container used directly
        else:
            image_container.markdown("**Image generation failed.**") # image_container used directly


def get_user_inputs(selected_word: str, all_word_details: dict) -> Optional[Dict]:
    """
    Function to validate and return word details based on user inputs.
    
    Args:
        selected_word (str): The word selected by the user.
        all_word_details (dict): Dictionary of all word details.
        
    Returns:
        Optional[Dict]: Details of the selected word, None if not present.
    """
    # Check if selected_word is in all_word_details
    if selected_word not in all_word_details:
        st.error(f"The word {selected_word} does not exist in the dictionary.")
        return None
    return all_word_details[selected_word]


def display_word_details_main(selected_word: str, all_word_details: dict, tts_settings: dict, sidebar) -> None:
    """
    Function to display the details of a selected word.

    Args:
        selected_word (str): The word selected by the user.
        all_word_details (dict): Dictionary of all word details.
        tts_settings (dict): Text-to-Speech settings.
        sidebar (Streamlit Sidebar): The sidebar object for output.
    """
    word_detail = get_user_inputs(selected_word, all_word_details)
    if word_detail is None:
        return

    # Display word details
    sidebar.markdown(f"<h3 style='color: crimson;'><span style='font-size: 28px; font-weight: bold;'>{selected_word}</span></h3>", unsafe_allow_html=True)  # changed to sidebar
    sidebar.markdown(f"<div style='color:CornflowerBlue'>Pronunciation guide: {word_detail.get('pronunciation', '')}</div>", unsafe_allow_html=True)  # changed to sidebar
    sidebar.markdown(f"Part of speech: {word_detail.get('part_of_speech', '')}")  # changed to sidebar
    
    # Display meanings
    meanings = word_detail.get('meanings', [])
    sidebar.markdown("Meanings:")  # changed to sidebar
    for meanings in meanings:
        sidebar.markdown(f"- {meanings}")  # changed to sidebar

    # Display example sentences
    example_sentences = word_detail.get('example', [])
    sidebar.markdown("Example of word used in sentence:")  # changed to sidebar
    for example_sentence in example_sentences:
        sidebar.markdown(f"- {example_sentence}")  # changed to sidebar

    # Display pronunciation
    sidebar.markdown("Listen to pronunciation:")  # changed to sidebar
    tts_service = tts_settings.get('tts_audio', 'gtts')
    audio = generate_audio(selected_word, tts_service)
    if os.path.isfile(audio):
        audio_file = open(audio, 'rb')
        audio_bytes = audio_file.read()
        sidebar.audio(audio_bytes, format='audio/wav')  # changed to sidebar
        os.remove(audio)  # Delete the temporary audio file after playing


def main_application(global_vars: dict) -> None:
    """
    Main application function.

    Args:
        global_vars (dict): A dictionary containing all global variables.
    """
    all_word_details = global_vars['all_word_details']
    tts_settings = global_vars['tts_settings']
    
    # Custom CSS for the image and title
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    similar_words = None
    selected_word = ""
    selected_word = ""

    # Define file path for "bnr1.png"
    image_path = os.path.join(os.getcwd(), "images", "bnr1.png")
    image = Image.open(image_path)
    # Center the image using Streamlit's layout feature
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=None, clamp=True, channels="RGB", output_format="PNG")

    # Instruction text
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div class="big-font">
                <span style="font-weight:bold; font-size:24px; color:CornflowerBlue;">Storyteller</span>
                <span style="color:white;">:</span>
                <span style="font-size:24px; color:crimson;">A'tugwewinu</span>
            </div>
            <ul style="text-align: left;">
                <li>Mi'kmaq storytelling plays a role in preserving the Mi'kmaq language.</li>
                <li>Storytelling is a vital aspect of the Mi'kmaq people's culture and tradition.</li>
                <li>It serves as a means to preserve and transmit knowledge, values, beliefs, and history from one generation to the next.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Initialize sidebar
    sidebar = st.sidebar

    # Load and display the image
    st.sidebar.image(image, use_column_width="auto", clamp=True, channels="RGB", output_format="png")

    # Instruction text
    st.sidebar.markdown("""
        <div style="text-align: center;">
            <div class="big-font -text center-text" style="color: ForestGreen; font-size: 24px;">Generate a Story</div>
        </div>
        <ul>
            <li>Type an <strong><span style="color: cornflowerblue;">English word</span></strong> into the <strong><span style="color: Ivory;">Search for a word</span></strong> field and press <strong><span style="color: Ivory;">Return</span></strong>.</li>
            <li>Select a <strong><span style="color: crimson;">Mi'kmaq word</span></strong> from the <strong><span style="color: Ivory;">Similar words</span></strong> dropdown list.</li>
            <li>Click <strong><span style="color: Ivory;">Generate Story</span></strong> and an audio and visual story will be created.</li>
        </ul>
    """, unsafe_allow_html=True)

    search_word = st.sidebar.text_input("Search for a word", value="")


    # Initialize the variable selected_word
    selected_word = None

    # Load word data
    if all_word_details is None:
        st.sidebar.error("Failed to load word data.")
        return

    # Generate word list and transform data to DataFrame
    word_list = list(all_word_details.keys())
    df = pd.DataFrame([(key, value["translation"], value["meanings"], value["example"]) for key, value in all_word_details.items()],
                        columns=["word", "translation", "meanings", "example"])

    # Sort the DataFrame by 'word' column in ascending order
    df = df.sort_values(by=['word'])

    # Resetting the index
    df = df.reset_index(drop=True)

    # Check if a search_word has been entered
    if search_word:  
        search_word_lower = search_word.lower()

        # Find the exact match in the "translation" field where the searched word is the first word
        exact_match_translation = df[df['translation'].apply(lambda translation: translation.lower().split()[0] == search_word_lower)]['word'].values

        if len(exact_match_translation) > 0:
            selected_word = exact_match_translation[0]
        else:
            # Find the word in the "meanings" field where the searched word is present
            similar_words_meanings = df[df['meanings'].apply(lambda meanings: any(search_word_lower in meanings.lower().split() for meanings in meanings))]['word'].values
            if len(similar_words_meanings) > 0:
                selected_word = similar_words_meanings[0]

        if not selected_word:
            st.sidebar.write("No similar word found.")
        
        if selected_word:
            # Get index of the selected word in dataframe
            selected_word_index = df.index[df['word'] == selected_word].tolist()[0]

            # Get next 19 words after the selected word
            next_words = df.iloc[selected_word_index+1 : selected_word_index+20]['word'].values

            # Combine the selected word with next 19 words
            combined_words = np.concatenate(([selected_word], next_words))

            selected_word = st.sidebar.selectbox("Similar words:", combined_words)

    # Display word details in sidebar
    if selected_word:
        display_word_details_main(selected_word, all_word_details, tts_settings, sidebar)

        # TTS service selection in the sidebar
        tts_service = sidebar.selectbox("Select a TTS service", ['gtts'], key="tts_service_selectbox", index=0)
        tts_settings["tts_audio"] = tts_service if tts_service else 'gtts'

        # Display selected word below submit button
        st.sidebar.markdown(f"Selected word: **{selected_word}**")
        
        # Submit button in the sidebar
        submit_button = sidebar.button(
            f"Generate Story about **{selected_word}**",
            help="Click to generate the story",
            key="submit_button",
            args=(selected_word, all_word_details, tts_service),
            kwargs={"generate_images": selected_word == "Submit with Images"},
            type="primary"
        )

        # Generate and display story, audio, and images
        if submit_button:
            st.info("Generating the story... This may take a minute. It's worth the wait!")  # <--- Display info message
            jsonl_file = "mikmaq_semantic.jsonl"
            themes = load_theme_and_story(jsonl_file)
            word_details = get_user_inputs(selected_word, all_word_details)
            if word_details is not None:
                meaning = word_details.get('meanings', [])[0] if word_details.get('meanings') else ""
                theme, story_word, image_theme = get_theme_and_story_word(themes, selected_word, meaning)
                story_text, _, _ = generate_story(word_details, theme, story_word, image_theme)

                try:
                    audio = generate_audio(story_text, tts_service)
                    display_story_and_audio(story_text, audio)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

                generate_and_display_images(story_text, image_theme)
                st.empty()  # <--- Removes info message

def main():
    """
    The main function of the application.
    """
    global_vars = load_env_variables_and_data()

    # You can use the global_vars dictionary to access the required values.
    api_keys = global_vars['api_keys']
    tts_settings = global_vars['tts_settings']
    local_data_files = global_vars['local_data_files']
    models = global_vars['models']

    all_word_details = global_vars['all_word_details']

    
    render_ui(CUSTOM_CSS)

    menu_options = {
        "Storyteller": lambda: main_application(global_vars),
    }

    selected_option = render_menu(list(menu_options.keys()))

    if selected_option and menu_options.get(selected_option):
        menu_options[selected_option]()


def render_ui(CUSTOM_CSS):
    """
    Renders the user interface components.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; color: Crimson; margin-top: -70px;">Lnu-AI</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">An Indigenous AI System</h3>', unsafe_allow_html=True)
    st.divider()


def render_menu(options: list) -> str:
    """
    Renders the menu with options.

    Args:
        options (list): A list of options to be displayed in the menu.

    Returns:
        str: The selected option from the menu.
    """
    icons = ["book", "chat", "puzzle fill", "archive"]
    return option_menu(None, options, icons=icons, menu_icon="cast", default_index=0, orientation="horizontal")


if __name__ == "__main__":
    main()