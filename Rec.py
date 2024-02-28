import streamlit as st
import pandas as pd
import json
from nltk.stem import PorterStemmer
import numpy as np
from  scipy.spatial.distance import cosine
from heapq import nsmallest,nlargest
from openai import OpenAI


data = pd.read_csv('my_dataframe.csv')

count_vector={}
with open('my_dict.json', 'r') as f:
        count_vector = json.load(f)

Term_set=[]

with open('Term_set.json', 'r') as f:
        Term_set = json.load(f)

def tokenize(txt):
  txt=str(txt)
  txt = txt.replace(';',' ')
  return txt.split()

def porter_stemmer(words):
  stemmer = PorterStemmer()
  return [stemmer.stem(word) for word in words]

def user_count_vector(doc): 
    lst=[]
    doc=tokenize(doc)
    doc=porter_stemmer(doc)   
    for j in Term_set:
         lst.append(doc.count(j))
    return lst
def top_similar_doc_cosine(count_vec,doc,k=3):
        lst={}
        for i in count_vec:
            lst[i]=1-cosine(count_vec[i],user_count_vector(doc))
        top_similar_doc = nlargest(k, lst, key = lst.get)
        return lst,top_similar_doc


def cosine_recommender(doc):
    # Read data from stdin
    

    user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
    user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
    rank=[]
    user=[]
    score=[]
    kw=[]
    count=1
    for i in rec_adv:
                   rank.append(count)
                   user.append(i)
                   score.append(user_score[i])
                   a=data[data['n']==i]['t']
                   for j in a:
                        k=" ".join(tokenize(j))
                   kw.append(k)
                   count+=1
    df = {
                                        'Ranking': rank,
                                        'Name': user,
                                        'Similarity Score': score,
                                        'Keywords': kw
                                    }
    data_str = json.dumps(df)
    
    #print("Done")
    with open('rec_result.json', 'w') as f:
        json.dump(df, f)
    return data_str




if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.clicked = True


st.title("Advisor Recommender System ")
name=st.text_input('Enter your Name')
keywords=st.text_area('Enter keywords of your reseach interest')

st.button('Submit', on_click=click_button)
data_dict={}
flag=0
if st.session_state.clicked:
    #with st.chat_message("Assistant"):
        st.session_state.clicked = False
        st.write("Hello "+name+"! Please wait while we retrieve some information.")
        import time
        with st.spinner(text='In progress'):
            output=cosine_recommender(keywords)           
            data_dict = json.loads(output)
            
            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = "gpt-3.5-turbo"
            if "flag" not in st.session_state:          
                st.session_state["flag"] = data_dict
                with open('rec_result.txt', 'w') as f:
                        msg="User name is "+ name+". User reseach interests are "+keywords+". Top 3 recommended advisor list:\n"
                        for i in range(len(data_dict['Ranking'])):
                            msg+=str(i+1)+'. name: '+ data_dict['Name'][i]
                            msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                            msg+='. Keywords: '+data_dict['Keywords'][i]+'\n'
                        f.write(msg)
                

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
                response="Welcome "+name+" to the Advisor Recommender System. Would you like an explaination of your recommendation for advisors?"
                st.session_state.messages.append({"role": "assistant", "content": response})
if "flag" in st.session_state:
    df = pd.DataFrame(st.session_state["flag"])
    st.dataframe(df)
    
            


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
if "messages"  in st.session_state:
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
                

