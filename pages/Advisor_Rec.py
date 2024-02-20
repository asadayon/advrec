from openai import OpenAI
import streamlit as st

st.title("Advisor Recommender System ChatBot")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    msg="" 
    with open('rec_result.txt', 'r') as f:
                    for line in f:
                        msg+=line
                  
    st.session_state.messages = [{'role':'system', 'content':msg+"""
You are an Advisor Recommendation System, you will explain why any advisor is recommended to the user. Recommendations were given based on the cosine similarity of user research interest and advisor research interest keywords. \
You are given a username and three recommended advisor names to the user. You are also provided with user research interests, and research interests of three recommended advisors. \
You may explain based on research interests are similar, and why. If the user asks about the cosine similarity score, which has a value of 0 to 1, you may use different examples to describe the similarity score.\
step-1: You first greet the user saying the user name is welcome to the Advisor Recommender system. Then ask 'Would you like a recommendation for an advisor?'\ 
Step 2: If positive, you will first provide the most similar advisor to the user, and provide a simple explanation why this advisor is recommended. \
Then you ask the user whether he or she wants to know about other recommended advisors or more explanations of why the previous advisor is recommended. \
step-3: If the user chooses another advisor, provide the next best-recommended advisor name and provide a simple explanation of why this advisor is recommended. \
If the user chooses to need more explanation, you will ask the user's background education. Based on the user background you explain the similarity score explaining their educational background. \
Then you ask the user whether he or she wants to know about other recommended advisors or more explanations about why the previous advisor is recommended. \
step-4: if positive repeat step-8, if the user has no query, thank the user and ask for feedback. \
You respond in a short, very conversational friendly style. 
"""}]

for message in st.session_state.messages:
    if message['role']=='system':
        continue
    if message['role']=='user':
        with st.chat_message(message["role"],avatar="👦"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user",avatar="👦"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

