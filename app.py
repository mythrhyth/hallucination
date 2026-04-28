# main_app.py
import streamlit as st
import json
import requests
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Hallucination Detection & Injection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoints
DETECTION_API = "http://127.0.0.1:8000/api/detection"
INJECTION_API = "http://127.0.0.1:8000"
INJECTION_ENDPOINTS = {
    "Heuristic": f"{INJECTION_API}/inject/heuristic",
    "Adversarial": f"{INJECTION_API}/inject/adversarial", 
    "Prompting": f"{INJECTION_API}/inject/prompting",
    "Sentiment-based": f"{INJECTION_API}/inject/sentiment"
}

# Session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Detection"

def go_to_injection():
    st.session_state.page = "Injection"

def go_to_detection():
    st.session_state.page = "Detection"

# Detection Page Functions
def detection_page():
    st.markdown('<h1 style="text-align:center;color:#1f77b4;">Hallucination Detection System</h1>', unsafe_allow_html=True)

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio(
        "Choose detection model:",
        ["LSTM with Embedding", "LSTM with Attention", "Recursive Hybrid Model"]
    )

    st.sidebar.info(
        f"Backend URL: {DETECTION_API}\n\nMake sure FastAPI is running."
    )

    # Input section
    st.subheader("Input Data")
    context = st.text_area("Context:", placeholder="Enter the problem context...", height=100)
    question = st.text_input("Question:", placeholder="Enter the question...")
    reasoning_text = st.text_area("Reasoning Steps:", placeholder="Step 1...\nStep 2...\nStep 3...", height=200)
    reasoning_steps = [s.strip() for s in reasoning_text.split('\n') if s.strip()]

    # Analyze button
    if st.button("Analyze for Hallucinations", type="primary", use_container_width=True):
        if not context or not question or not reasoning_steps:
            st.error("Please provide context, question, and reasoning steps.")
            return
        
        payload = {"context": context, "question": question, "reasoning_steps": reasoning_steps}
        if model_choice == "LSTM with Embedding":
            endpoint = f"{DETECTION_API}/lstm_emb/predict"
        elif model_choice == "LSTM with Attention":
            endpoint = f"{DETECTION_API}/lstm_att/predict"
        else:
            endpoint = f"{DETECTION_API}/recursive_hybrid/predict"

        try:
            with st.spinner(f"Running {model_choice}..."):
                response = requests.post(endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                display_detection_results(result)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")

    # Navigation to injection page
    st.markdown("---")
    st.info("Want to inject hallucinations instead?")
    if st.button("Go to Injection System"):
        go_to_injection()

def display_detection_results(result):
    st.success("Analysis Complete!")
    st.subheader("Results Summary")

    context = result.get("context")
    question = result.get("question")
    results = result.get("results", [])

    st.write(f"**Context:** {context}")
    st.write(f"**Question:** {question}")
    st.write("### Step Analysis:")

    hallucinated_steps = 0
    for idx, step_data in enumerate(results, start=1):
        step = step_data["step"]
        prediction = step_data["prediction"]
        confidence = step_data["confidence"]

        if prediction == "inconsistent":
            hallucinated_steps += 1
            st.markdown(f"""
            <div style="border-left:4px solid #ffc107;padding:8px;border-radius:5px;color:white;margin:4px 0;">
                <b>Step {idx} (Inconsistent)</b><br>{step}<br>
                <i>Confidence: {confidence}</i>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="border-left:4px solid #28a745;padding:8px;border-radius:5px;color:white;margin:4px 0;">
                <b>Step {idx} (Consistent)</b><br>{step}<br>
                <i>Confidence: {confidence}</i>
            </div>
            """, unsafe_allow_html=True)

    total_steps = len(results)
    if hallucinated_steps == 0:
        st.success("All reasoning steps are consistent — no hallucinations detected!")
    else:
        st.warning(f"{hallucinated_steps} out of {total_steps} steps are hallucinated.")

    # Pie chart
    if total_steps > 0:
        labels = ["Consistent", "Inconsistent"]
        values = [total_steps - hallucinated_steps, hallucinated_steps]
        colors = ['#28a745', '#ffc107']
        
        fig = px.pie(
            values=values, 
            names=labels, 
            title="Consistency Distribution",
            color=labels,
            color_discrete_map={"Consistent": "#28a745", "Inconsistent": "#ffc107"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Injection Page Functions
def injection_page():
    st.markdown('<h1 style="text-align:center;color:#e83e8c;">Hallucination Injection System</h1>', unsafe_allow_html=True)

    st.sidebar.header("Injection Configuration")
    model_type = st.sidebar.selectbox("Choose injection method", list(INJECTION_ENDPOINTS.keys()))
    
    mode = st.sidebar.selectbox("Injection Mode", ["single", "comprehensive"], 
                               help="Single: One modified version | Comprehensive: Multiple variants")
    
    # Show additional parameters based on model type
    if model_type == "Adversarial":
        strategy = st.sidebar.selectbox("Adversarial Strategy", ["random", "numbers_first", "operations_first"])
        max_edits = st.sidebar.slider("Max edits per step", 1, 5, 1)
        if mode == "comprehensive":
            num_variants = st.sidebar.slider("Comprehensive variants", 1, 10, 5)
        else:
            num_variants = 1
    elif model_type == "Sentiment-based":
        num_steps_to_alter = st.sidebar.slider("Number of steps to alter", 1, 5, 1)
    else:
        strategy, max_edits, num_variants, num_steps_to_alter = None, None, None, None

    st.sidebar.info(
        f"Backend URL: {INJECTION_API}\n\nMake sure FastAPI is running."
    )

    # Input section - similar to detection page
    st.subheader("Input Data")
    context = st.text_area("Context:", placeholder="Enter the problem context...", height=100)
    question = st.text_input("Question:", placeholder="Enter the question...")
    reasoning_text = st.text_area("Reasoning Steps:", placeholder="Step 1...\nStep 2...\nStep 3...", height=200)
    reasoning_steps = [s.strip() for s in reasoning_text.split('\n') if s.strip()]

    # Method descriptions
    st.markdown("""
    ### Injection Methods:
    - **Heuristic**: Rule-based modifications using patterns and templates
    - **Adversarial**: Text-level edits with strategic modifications  
    - **Prompting**: LLM-style hallucination generation
    - **Sentiment-based**: Flip reasoning tone or sentiment
    """)

    if st.button("Inject Hallucinations", type="primary", use_container_width=True):
        if not context or not question or not reasoning_steps:
            st.error("Please provide context, question, and reasoning steps.")
            return
        
        try:
            # Prepare payload for injection API
            input_data = {
                "context": context,
                "question": question, 
                "reasoning_steps": reasoning_steps
            }
            
            # Build payload based on model type
            payload = {"data": input_data, "mode": mode, "pure": False}

            if model_type == "Adversarial":
                payload.update({
                    "strategy": strategy,
                    "max_edits": max_edits,
                    "num_variants": num_variants
                })
            elif model_type == "Sentiment-based":
                payload.update({
                    "num_steps_to_alter": num_steps_to_alter
                })

            with st.spinner(f"Running {model_type} injection..."):
                response = requests.post(INJECTION_ENDPOINTS[model_type], json=payload)

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    display_injection_results(result, model_type, mode, input_data)
                else:
                    st.error(f"Injection failed: {result.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Error during injection: {str(e)}")

    # Navigation back to detection
    st.markdown("---")
    st.info("Want to detect hallucinations instead?")
    if st.button("Go to Detection System"):
        go_to_detection()

def display_injection_results(result, model_type, mode, original_input):
    st.success(f"{model_type} Injection Completed!")
    
    # Display original input
    st.subheader("Original Input")
    st.write(f"**Context:** {original_input['context']}")
    st.write(f"**Question:** {original_input['question']}")
    
    st.write("**Original Reasoning Steps:**")
    for idx, step in enumerate(original_input['reasoning_steps'], 1):
        st.markdown(f"""
        <div style="border-left:4px solid #1f77b4;padding:8px;border-radius:5px;margin:4px 0;">
            <b>Step {idx}</b><br>{step}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display injection results - handle different response formats
    injection_data = result.get("result", {})
    
    # Handle prompting pipeline specifically - it returns output wrapped in input/output format
    if model_type == "Prompting" and "output" in injection_data:
        injection_data = injection_data["output"]
    
    # Handle different pipeline response formats
    if isinstance(injection_data, list):
        # Heuristic comprehensive mode returns a list
        if mode == "comprehensive":
            display_comprehensive_injection(injection_data, model_type, original_input)
        else:
            # If single mode but got list, take first item
            if injection_data:
                display_single_injection(injection_data[0], model_type, original_input)
            else:
                st.error("No injection results found")
    elif isinstance(injection_data, dict):
        if "modified_reasoning" in injection_data or "reasoning_steps" in injection_data:
            display_single_injection(injection_data, model_type, original_input)
        elif "variants" in injection_data:
            display_comprehensive_injection(injection_data.get("variants", []), model_type, original_input)
        else:
            # If we have a dict but no expected keys, check if it's the prompting format
            if "reasoning_steps" in injection_data and "step_labels" in injection_data:
                display_single_injection(injection_data, model_type, original_input)
            else:
                st.warning("Unexpected response format from injection API")
                st.json(result)
    else:
        st.warning("Unexpected response format from injection API")
        st.json(result)

def display_single_injection(injection_data, model_type, original_input):
    st.subheader("Modified Reasoning Steps")
    
    # Extract modified steps based on different pipeline formats
    if "modified_reasoning" in injection_data:
        modified_steps = injection_data.get("modified_reasoning", [])
    elif "reasoning_steps" in injection_data:
        modified_steps = injection_data.get("reasoning_steps", [])
    else:
        st.error("No modified steps found in response")
        return
    
    original_steps = original_input['reasoning_steps']
    
    if not modified_steps or len(modified_steps) != len(original_steps):
        st.error("Modified steps don't match original steps length")
        return
    
    # Create comparison table
    st.write("### Step-by-Step Comparison")
    
    modified_count = 0
    for idx, (orig, mod) in enumerate(zip(original_steps, modified_steps), 1):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="border-left:4px solid #1f77b4;padding:8px;border-radius:5px;margin:4px 0;">
                <b>Original Step {idx}</b><br>{orig}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if orig != mod:
                modified_count += 1
                st.markdown(f"""
                <div style="border-left:4px solid #e83e8c;padding:8px;border-radius:5px;margin:4px 0;">
                    <b>Modified Step {idx}</b><br>{mod}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="border-left:4px solid #2ecc71;padding:8px;border-radius:5px;margin:4px 0;">
                    <b>Modified Step {idx}</b><br>{mod}
                </div>
                """, unsafe_allow_html=True)
    
    # Statistics
    total_steps = len(original_steps)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Steps", total_steps)
    with col2:
        st.metric("Steps Modified", modified_count)
    with col3:
        modification_rate = (modified_count / total_steps) * 100 if total_steps > 0 else 0
        st.metric("Modification Rate", f"{modification_rate:.1f}%")
    
    # Show injection details if available
    if "_injection_info" in injection_data:
        with st.expander("Injection Details"):
            info = injection_data["_injection_info"]
            st.write(f"**Rule Applied:** {info.get('rule', 'N/A')}")
            st.write(f"**Step Index:** {info.get('step_index', 'N/A')}")
            st.write("**Original Step:**")
            st.info(info.get('original', 'N/A'))
            st.write("**Modified Step:**")
            st.error(info.get('modified', 'N/A'))
    
    # Show step labels if available
    if "step_labels" in injection_data:
        st.write("### Step Labels (1 = Modified, 0 = Original)")
        labels = injection_data["step_labels"]
        for idx, label in enumerate(labels, 1):
            status = "Modified" if label == 1 else "Original"
            st.write(f"Step {idx}: {status}")

def display_comprehensive_injection(variants, model_type, original_input):
    st.subheader("Comprehensive Injection Results")
    
    if not variants:
        st.error("No variants found in response")
        return
    
    # Show statistics for all variants
    st.write("### Variants Overview")
    
    cols = st.columns(min(len(variants), 5))  # Max 5 columns
    for idx, variant in enumerate(variants):
        if idx < len(cols):
            with cols[idx]:
                # Calculate modification count
                if "step_labels" in variant:
                    mod_count = sum(variant["step_labels"])
                else:
                    # Estimate by comparing steps
                    original_steps = original_input['reasoning_steps']
                    modified_steps = variant.get("reasoning_steps", variant.get("modified_reasoning", []))
                    mod_count = sum(1 for orig, mod in zip(original_steps, modified_steps) if orig != mod)
                
                total_steps = len(original_input['reasoning_steps'])
                st.metric(f"Variant {idx+1}", f"{mod_count}/{total_steps} modified")
    
    st.markdown("---")
    
    # Display each variant
    for variant_idx, variant in enumerate(variants, 1):
        st.subheader(f"Variant {variant_idx}")
        
        # Extract modified steps
        if "modified_reasoning" in variant:
            modified_steps = variant.get("modified_reasoning", [])
        elif "reasoning_steps" in variant:
            modified_steps = variant.get("reasoning_steps", [])
        else:
            continue
            
        original_steps = original_input['reasoning_steps']
        
        # Calculate modification count
        if "step_labels" in variant:
            modification_count = sum(variant["step_labels"])
        else:
            modification_count = sum(1 for orig, mod in zip(original_steps, modified_steps) if orig != mod)
        
        st.write(f"**Modifications:** {modification_count} steps changed")
        
        # Show step-by-step comparison for this variant
        for step_idx, (orig, mod) in enumerate(zip(original_steps, modified_steps), 1):
            col1, col2 = st.columns(2)
            
            with col1:
                if step_idx == 1:  # Label only once per variant
                    st.markdown("**Original Steps**")
                st.markdown(f"""
                <div style="padding:6px;border-radius:3px;margin:2px 0;">
                    {orig}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if step_idx == 1:  # Label only once per variant
                    st.markdown("**Modified Steps**")
                if orig != mod:
                    st.markdown(f"""
                    <div style="padding:6px;border-radius:3px;margin:2px 0;">
                        {mod}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding:6px;border-radius:3px;margin:2px 0;">
                        {mod}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show injection details if available
        if "_injection_info" in variant:
            with st.expander(f"Injection Details for Variant {variant_idx}"):
                info = variant["_injection_info"]
                st.write(f"**Rule:** {info.get('rule', 'N/A')}")
                st.write(f"**Step Index:** {info.get('step_index', 'N/A')}")
                st.write("**Original:**")
                st.info(info.get('original', 'N/A'))
                st.write("**Modified:**")
                st.error(info.get('modified', 'N/A'))
        
        st.markdown("---")

# Main app logic
if st.session_state.page == "Detection":
    detection_page()
else:
    injection_page()