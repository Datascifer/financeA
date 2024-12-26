import os
import openai
from dotenv import load_dotenv
import panel as pn

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("API_KEY")  # Securely access the API key

# Initialize Panel
pn.extension()

# Helper Functions
def get_completion_from_messages(messages, model="gpt-4", temperature=0.7):
    """
    Get a response using a conversation history.
    
    Parameters:
        - messages (list): List of conversation messages (system, user, assistant).
        - model (str): OpenAI model to use (default is gpt-4).
        - temperature (float): Creativity level (0.0 to 1.0).

    Returns:
        - str: Assistant's response.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message["content"]
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"


# Global Variables
context = [  # Initial system message
    {
        'role': 'system',
        'content': """
You are JokeBot, a virtual assistant for comedians to find and create jokes. 
You specialize in adapting to different comedic styles:
- Norm Macdonald: Dry, deadpan, and absurd.
- Richard Pryor: Raw, energetic, observational humor.
- Betty White: Charming, cheeky, and witty.
- Robin Williams: Energetic, improvisational, and witty storytelling.

Guide users to select their preferred comedic style and joke theme (e.g., work, food, relationships). 
Deliver jokes that match their selections and offer alternative punchlines if asked. Keep the tone engaging and witty.
"""
    }
]

conversation_display = pn.Column()  # Container for conversation messages
input_boxes = []  # List to hold all input boxes

# Dropdowns for Comedy Style and Theme
style_dropdown = pn.widgets.Select(
    name="Comedy Style", 
    options=["Select A Style", "Norm Macdonald", "Richard Pryor", "Betty White", "Robin Williams"], 
    value="Select A Style"
)

theme_dropdown = pn.widgets.Select(
    name="Theme", 
    options=["Select A Style", "Technology", "Data Science", "Politics", "Best Worst Advice"], 
    value="Select A Style"
)

temperature_slider = pn.widgets.FloatSlider(
    name="Temperature (Creativity)", start=0.0, end=1.0, step=0.1, value=0.7
)

# Function to dynamically add a new input box and display conversation
def collect_messages(event):
    """
    Collect user input, process the conversation, and dynamically add input boxes.
    """
    # Get dropdown values
    selected_style = style_dropdown.value
    selected_theme = theme_dropdown.value
    selected_temperature = temperature_slider.value

    # Validation: Check if a style is selected
    if selected_style == "Select A Style":
        conversation_display.append(
            pn.pane.HTML(
                "<div style='background-color: #FFE5E5; padding: 10px; border-radius: 5px;'>"
                "<b>Error:</b> Please select a valid comedy style before generating a joke.</div>"
            )
        )
        return

    # Generate user input based on dropdown selections
    user_input = f"I'd like a joke in the style of {selected_style} about {selected_theme}."
    context.append({'role': 'user', 'content': user_input})

    # Add typing indicator
    conversation_display.append(
        pn.pane.HTML(
            "<div style='background-color: #EFEFEF; padding: 10px; border-radius: 5px;'>"
            "<i>JokeBot is thinking...</i></div>"
        )
    )

    # Get assistant response
    response = get_completion_from_messages(context, temperature=selected_temperature)
    
    # Remove typing indicator and display response
    conversation_display.pop(-1)
    conversation_display.append(
        pn.pane.HTML(
            f"<div style='background-color: #DFF6FF; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>"
            f"<b>User:</b> {user_input}</div>"
        )
    )
    conversation_display.append(
        pn.pane.HTML(
            f"<div style='background-color: #F6F6F6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
            f"<b>Assistant:</b> {response} <br><br><b>Want another joke? Change the style or theme and press the button!</b></div>"
        )
    )


# Button to Trigger Joke Generation
generate_joke_button = pn.widgets.Button(name="Generate Joke", button_type="primary")
generate_joke_button.on_click(collect_messages)

# Initial Assistant Introduction and Instructions
conversation_display.append(
    pn.pane.HTML(
        f"<div style='background-color: #F6F6F6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
        f"<b>Assistant:</b> Hi there! I'm JokeBot, your virtual assistant for renting jokes. "
        f"Select your preferred comedic style and theme using the dropdowns below, adjust the creativity with the slider, "
        f"and press <b>Generate Joke</b>. Let's get started!</div>"
    )
)

# Layout with Dropdowns, Button, and Conversation Display
dashboard = pn.Column(
    pn.pane.Markdown("# JokeBot: Rent a Joke for Comedians"),
    style_dropdown,
    theme_dropdown,
    temperature_slider,
    generate_joke_button,
    conversation_display
)

# Serve the dashboard
dashboard.servable()
