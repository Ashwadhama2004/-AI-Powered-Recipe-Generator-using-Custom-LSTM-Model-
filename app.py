import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st

# Paths to model and tokenizer
MODEL_PATH = r"D:\mL project\AI based Recipie\recipe_generator_model.h5"
TOKENIZER_PATH = r"D:\mL project\AI based Recipie\recipe_tokenizer.pickle"

# Load trained model
model = load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set max input length and vocab size (can be retrieved from the tokenizer)
max_input_length = 100  # Set this based on your model or tokenizer
VOCAB_SIZE = len(tokenizer.word_index) + 1  # Vocabulary size based on the tokenizer

# Clean the input text (modify this function as needed)
def clean_text(text):
    text = text.lower()
    text = text.strip()
    return text

# Recipe generation function
def generate_recipe(input_text, encoder_model, decoder_model, tokenizer, max_length=150):
    # Clean and format input
    if not input_text.startswith("ingredients:"):
        input_text = "ingredients: " + clean_text(input_text)
    
    # Convert to sequence
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_length, padding="post")
    
    # Get states from encoder
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Start with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get("startseq", 1)
    
    # Track output
    decoded_sentence = []
    stop_condition = False
    
    while not stop_condition:
        # Predict next token
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Get the token with highest probability
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        
        # Convert back to word
        if sampled_token_index > 0 and sampled_token_index < VOCAB_SIZE:
            sampled_word = None
            for word, index in tokenizer.word_index.items():
                if index == sampled_token_index:
                    sampled_word = word
                    break
        else:
            sampled_word = ""
        
        # Stop conditions
        if sampled_word == "endseq" or len(decoded_sentence) > max_length:
            stop_condition = True
        
        decoded_sentence.append(sampled_word)
        
        # Update target sequence
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return " ".join(decoded_sentence)


# Streamlit UI for recipe generation
st.title("Recipe Generator")

# Input box for the user to enter ingredients
input_text = st.text_area("Enter the ingredients:")

if st.button("Generate Recipe"):
    if input_text:
        # Generate the recipe
        recipe = generate_recipe(input_text, model, model, tokenizer)
        
        # Display the generated recipe
        st.write(f"Generated Recipe:\n{recipe}")
    else:
        st.write("Please enter some ingredients to generate a recipe.")
