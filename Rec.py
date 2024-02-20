import streamlit as st
import pandas as pd
import json
from nltk.stem import PorterStemmer
import numpy as np
from  scipy.spatial.distance import cosine
from heapq import nsmallest,nlargest



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
    st.session_state.clicked = True

st.title("Advisor Recommender System ")
name=st.text_input('Enter your Name')
keywords=st.text_area('Enter keywords of your reseach interest')

st.button('Submit', on_click=click_button)

if st.session_state.clicked:
    with st.chat_message("Assistant"):
        st.write("Hello "+name+"! Please wait while we retrieve some information.")
        import time
        with st.spinner(text='In progress'):
            #process = subprocess.Popen(['python', 'TFIDF_recommender.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            #output, error = process.communicate(input=keywords)
            output=cosine_recommender(keywords)
            data_dict={}
            data_dict = json.loads(output)
                
            df = pd.DataFrame(data_dict)
            st.dataframe(df)
            with open('rec_result.txt', 'w') as f:
                    msg="User name is "+ name+". User reseach interests are "+keywords+". Top 3 recommended advisor list:\n"
                    for i in range(len(data_dict['Ranking'])):
                        msg+=str(i)+'. name: '+ data_dict['Name'][i]
                        msg+='. Similarity score: '+str(data_dict['Similarity Score'][i])
                        msg+='. Keywords: '+data_dict['Keywords'][i]+'\n'
                    f.write(msg)

           


#def main_page():
#    st.markdown("# Main page 🎈")
#    st.sidebar.markdown("# Main page 🎈")

#def page2():
#    st.markdown("# Page 2 ❄️")
#    st.sidebar.markdown("# Page 2 ❄️")

#def page3():
#    st.markdown("# Page 3 🎉")
#    st.sidebar.markdown("# Page 3 🎉")


#selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
#Advisor_Rec()

