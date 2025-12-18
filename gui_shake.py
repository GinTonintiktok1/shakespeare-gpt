# ========================================
# shakespeare_app.py - STREAMLIT GUI
# ========================================

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import torch
from shake_py import GPT, generate
import time
from datetime import datetime

# ========================================
# CONFIGURAZIONE PAGINA
# ========================================
st.set_page_config(
    page_title="Shakespeare GPT",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# STILE CSS CUSTOM
# ========================================
st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .generated-text {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Georgia', serif;
        font-size: 16px;
        line-height: 1.8;
        color: #e0e0e0;
        border-left: 4px solid #9c27b0;
    }
    .character-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# CARICAMENTO MODELLO (cached!)
# ========================================
@st.cache_resource
def load_model():
    device = 'cpu'
    model = GPT(256, 128, 8, 6, 512, 128, 0.1).to(device)
    model.load_state_dict(torch.load('shakespeare_gpt_production.pth', map_location='cpu'))
    model.eval()
    return model, device

# ========================================
# PERSONAGGI SHAKESPEARE
# ========================================
CHARACTERS = {
    "🗡️ Romeo": {
        "prompt": "ROMEO:",
        "description": "Romantico e passionale protagonista",
        "default_temp": 0.8,
        "emoji": "❤️"
    },
    "🌹 Juliet": {
        "prompt": "JULIET:",
        "description": "Giovane e ardente innamorata",
        "default_temp": 0.9,
        "emoji": "💕"
    },
    "💀 Hamlet": {
        "prompt": "HAMLET:",
        "description": "Principe tormentato e filosofico",
        "default_temp": 0.7,
        "emoji": "🤔"
    },
    "👑 King Lear": {
        "prompt": "KING LEAR:",
        "description": "Re tragico e potente",
        "default_temp": 0.75,
        "emoji": "👑"
    },
    "⚔️ Macbeth": {
        "prompt": "MACBETH:",
        "description": "Ambizioso e tormentato",
        "default_temp": 0.8,
        "emoji": "🗡️"
    },
    "🌙 Ophelia": {
        "prompt": "OPHELIA:",
        "description": "Delicata e tragica",
        "default_temp": 0.85,
        "emoji": "🌸"
    },
    "🎭 Mercutio": {
        "prompt": "MERCUTIO:",
        "description": "Spiritoso e sarcastico",
        "default_temp": 1.0,
        "emoji": "😄"
    },
    "👻 Ghost": {
        "prompt": "GHOST:",
        "description": "Spettrale e misterioso",
        "default_temp": 0.6,
        "emoji": "👻"
    },
    "🧙 Prospero": {
        "prompt": "PROSPERO:",
        "description": "Mago saggio e potente",
        "default_temp": 0.7,
        "emoji": "✨"
    },
    "✍️ Custom": {
        "prompt": "",
        "description": "Inserisci il tuo prompt personalizzato",
        "default_temp": 0.8,
        "emoji": "📝"
    }
}

# ========================================
# INIZIALIZZAZIONE SESSION STATE
# ========================================
if 'history' not in st.session_state:
    st.session_state.history = []

if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""

# ========================================
# HEADER
# ========================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🎭 Shakespeare GPT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Generate authentic Shakespeare-style text with AI</p>", unsafe_allow_html=True)

st.divider()

# ========================================
# CARICA MODELLO
# ========================================
with st.spinner("🎨 Loading AI model..."):
    try:
        model, device = load_model()
        st.success("✅ Model loaded! (1.26M parameters)")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ========================================
# SIDEBAR - CONTROLLI
# ========================================
with st.sidebar:
    st.markdown("### 🎨 Generation Settings")
    
    # Selezione personaggio
    st.markdown("#### 🎭 Choose Character")
    selected_character = st.selectbox(
        "Character",
        options=list(CHARACTERS.keys()),
        label_visibility="collapsed"
    )
    
    # Info personaggio
    char_info = CHARACTERS[selected_character]
    st.markdown(f"""
        <div class='character-card'>
            <h4>{char_info['emoji']} {selected_character.split(' ')[1]}</h4>
            <p style='margin: 0; font-size: 14px;'>{char_info['description']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Prompt custom
    if selected_character == "✍️ Custom":
        custom_prompt = st.text_input(
            "Enter custom prompt:",
            value="",
            placeholder="e.g., PUCK:"
        )
        prompt = custom_prompt if custom_prompt else "ROMEO:"
    else:
        prompt = char_info['prompt']
    
    st.divider()
    
    # Temperature
    st.markdown("#### 🌡️ Temperature")
    st.markdown("*Controls creativity (higher = more creative)*")
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=char_info['default_temp'],
        step=0.1,
        label_visibility="collapsed"
    )
    
    # Length
    st.markdown("#### 📏 Text Length")
    max_length = st.slider(
        "Maximum characters",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
        label_visibility="collapsed"
    )
    
    # Top-k
    with st.expander("⚙️ Advanced Settings"):
        top_k = st.slider("Top-K sampling", 10, 100, 40, 5)
        show_metadata = st.checkbox("Show generation metadata", value=True)
    
    st.divider()
    
    # Bottone generazione
    generate_btn = st.button(
        "🎬 Generate Shakespeare",
        type="primary",
        use_container_width=True
    )
    
    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.generated_text = ""
        st.rerun()

# ========================================
# MAIN AREA
# ========================================

# Tabs
tab1, tab2, tab3 = st.tabs(["📜 Generate", "📚 History", "ℹ️ About"])

with tab1:
    # Area generazione
    if generate_btn:
        if not prompt or prompt.strip() == "":
            st.warning("⚠️ Please enter a prompt!")
        else:
            with st.spinner(f"✨ {selected_character} is speaking..."):
                start_time = time.time()
                
                try:
                    # Genera testo
                    generated = generate(
                        model, 
                        prompt, 
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_k=top_k
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # Salva in session state
                    st.session_state.generated_text = generated
                    
                    # Aggiungi a history
                    st.session_state.history.append({
                        'character': selected_character,
                        'prompt': prompt,
                        'text': generated,
                        'temperature': temperature,
                        'length': len(generated),
                        'time': elapsed,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.success(f"✅ Generated in {elapsed:.2f}s!")
                    
                except Exception as e:
                    st.error(f"❌ Generation error: {e}")
    
    # Mostra testo generato
    if st.session_state.generated_text:
        st.markdown("### 📜 Generated Text")
        
        # Mostra in box stilizzato
        st.markdown(
            f"<div class='generated-text'>{st.session_state.generated_text}</div>",
            unsafe_allow_html=True
        )
        
        # Metadata
        if show_metadata and st.session_state.history:
            last_gen = st.session_state.history[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Character", last_gen['character'].split()[1])
            with col2:
                st.metric("Temperature", f"{last_gen['temperature']:.1f}")
            with col3:
                st.metric("Length", f"{last_gen['length']} chars")
            with col4:
                st.metric("Time", f"{last_gen['time']:.2f}s")
        
        # Azioni
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                "📥 Download",
                data=st.session_state.generated_text,
                file_name=f"shakespeare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.info("💡 Select text and press Ctrl+C to copy")
        
    else:
        # Placeholder
        st.info("👆 Select a character and click 'Generate Shakespeare' to start!")
        
        # Examples
        st.markdown("### 💡 Quick Examples")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("❤️ Romeo's Love", use_container_width=True):
                st.session_state.quick_prompt = "ROMEO:"
                st.rerun()
        
        with col2:
            if st.button("🤔 Hamlet's Thoughts", use_container_width=True):
                st.session_state.quick_prompt = "HAMLET:"
                st.rerun()
        
        with col3:
            if st.button("👑 Lear's Wrath", use_container_width=True):
                st.session_state.quick_prompt = "KING LEAR:"
                st.rerun()

with tab2:
    # History
    st.markdown("### 📚 Generation History")
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{item['timestamp']} - {item['character']} ({item['length']} chars)"):
                st.markdown(f"**Prompt:** `{item['prompt']}`")
                st.markdown(f"**Settings:** Temp={item['temperature']:.1f}, Time={item['time']:.2f}s")
                st.divider()
                st.text_area(
                    "Generated text:",
                    value=item['text'],
                    height=200,
                    key=f"history_{i}",
                    label_visibility="collapsed"
                )
                st.download_button(
                    "📥 Download this",
                    data=item['text'],
                    file_name=f"shakespeare_{item['timestamp'].replace(':', '')}.txt",
                    key=f"download_{i}"
                )
    else:
        st.info("No generations yet. Start creating!")

with tab3:
    # About
    st.markdown("""
    ### 🎭 About Shakespeare GPT
    
    This AI model was trained on Shakespeare's complete works and can generate 
    authentic-sounding Shakespearean text in the style of different characters.
    
    **🧠 Model Architecture:**
    - Transformer-based GPT architecture
    - 1.26 Million parameters
    - 6 layers, 8 attention heads
    - Character-level tokenization
    
    **📊 Training:**
    - Dataset: Shakespeare Complete Works
    - Training Loss: 1.1187
    - Validation Loss: 1.5050
    - Epochs: 5
    
    **🎨 Features:**
    - 9 pre-configured characters
    - Adjustable creativity (temperature)
    - Custom prompt support
    - Generation history
    - Download outputs
    
    **💡 Tips for Best Results:**
    - Use lower temperature (0.6-0.7) for coherent, serious text
    - Use higher temperature (0.9-1.2) for creative, experimental text
    - Start with character names for dialogue
    - Try different characters for different styles!
    
    ---
    
    Made with ❤️ using PyTorch & Streamlit
    """)

# ========================================
# FOOTER
# ========================================
st.divider()
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 12px;'>"
        "Powered by Shakespeare GPT (1.26M params) | PyTorch 2.x"
        "</p>",
        unsafe_allow_html=True
    )
