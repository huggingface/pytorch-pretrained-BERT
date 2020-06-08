import time
import pyarrow
from eli5_utils import *

import streamlit as st

MODEL_TYPE='bart'
LOAD_DENSE_INDEX=True

@st.cache(allow_output_mutation=True)
def load_models():
    if LOAD_DENSE_INDEX:
        qar_tokenizer, qar_model = make_qa_retriever_model(
            model_name="google/bert_uncased_L-8_H-768_A-12",
            from_file="retriever_models/eli5_retriever_model_l-8_h-768_b-512-512_9.pth",
            device="cuda:0"
        )
    else:
        qar_tokenizer, qar_model = (None, None)
    if MODEL_TYPE == 'bart':
        s2s_tokenizer, s2s_model = make_qa_s2s_model(
            model_name="bart-large",
            from_file="seq2seq_models/eli5_bart_model_512_2.pth",
            device="cuda:0"
        )
    else:
        s2s_tokenizer, s2s_model = make_qa_s2s_model(
            model_name="t5-small",
            from_file="seq2seq_models/eli5_t5_model_1024_4.pth",
            device="cuda:0"
        )
    return (qar_tokenizer, qar_model, s2s_tokenizer, s2s_model)

@st.cache(allow_output_mutation=True)
def load_indexes():
    if LOAD_DENSE_INDEX:
        faiss_res = faiss.StandardGpuResources()
        wiki40b_passages = nlp.load_dataset(path="/home/yacine/Code/nlp/datasets/wiki_snippets", name="wiki40b_en_100_0")['train']
        wiki40b_passage_reps = np.memmap(
            'wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat',
            dtype='float32', mode='r',
            shape=(wiki40b_passages.num_rows, 128)
        )
        wiki40b_index_flat = faiss.IndexFlatIP(128)
        wiki40b_gpu_index_flat = faiss.index_cpu_to_gpu(faiss_res, 1, wiki40b_index_flat)
        wiki40b_gpu_index_flat.add(wiki40b_passage_reps) # TODO fix for larger GPU
    else:
        wiki40b_passages, wiki40b_gpu_index_flat = (None, None)
    es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    return (wiki40b_passages, wiki40b_gpu_index_flat, es_client)

@st.cache(allow_output_mutation=True)
def load_train_data():
    eli5 = nlp.load_dataset("/home/yacine/Code/nlp/datasets/explainlikeimfive", name="LFQA_reddit")
    eli5_train = eli5['train_eli5']
    eli5_train_q_reps = np.memmap(
        'eli5_questions_reps.dat',
        dtype='float32', mode='r',
        shape=(eli5_train.num_rows, 128)
    )
    eli5_train_q_index = faiss.IndexFlatIP(128)
    eli5_train_q_index.add(eli5_train_q_reps)
    return (eli5_train, eli5_train_q_index)

passages, gpu_dense_index, es_client = load_indexes()
qar_tokenizer, qar_model, s2s_tokenizer, s2s_model = load_models()
eli5_train, eli5_train_q_index = load_train_data()

def find_nearest_training(question, n_results=10):
    q_rep = embed_questions_for_retrieval([question], qar_tokenizer, qar_model)
    D, I = eli5_train_q_index.search(q_rep, n_results)
    nn_examples = [eli5_train[int(i)] for i in I[0]]
    return nn_examples

def make_support(question, source='wiki40b', method='dense', n_results=10):
    if source == 'none':
        support_doc, hit_lst = (' <P> '.join(['' for _ in range(11)]).strip(), [])
    else:
        if method == 'dense':
            support_doc, hit_lst = query_qa_dense_index(
                question,
                qar_model, qar_tokenizer,
                passages, gpu_dense_index,
                n_results
            )
        else:
            support_doc, hit_lst = query_es_index(
                question,
                es_client,
                index_name='english_wiki40b_snippets_100w',
                n_results=n_results,
            )
    support_list = [(res['article_title'], res['section_title'].strip(), res['score'], res['passage_text']) for res in hit_lst]
    question_doc = "question: {} context: {}".format(question, support_doc)
    return question_doc, support_list


# @st.cache(allow_output_mutation=True)
# @st.cache(allow_output_mutation=True, hash_funcs={torch.Tensor: (lambda _ : None)})
def answer_question(question_doc,
                    min_len=64, max_len=256,
                    sampling=False,
                    n_beams=4,
                    top_p=0.95, temp=0.8):
    answer = qa_s2s_generate(
        question_doc, s2s_model, s2s_tokenizer,
        num_answers=1,
        num_beams=n_beams,
        min_len=min_len,
        max_len=max_len,
        do_sample=sampling,
        temp=temp,
        top_p=top_p,
        top_k=None,
        max_input_length=512,
        device="cuda:0"
    )[0]
    return (answer, support_list)


st.title('Long Form Question Answering with ELI5')

# Start sidebar
import base64
from pathlib import Path

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/jpg;base64,{}'>".format(
    img_to_bytes("images/huggingface_logo.jpg")
)
header_full = """
<html>
  <head>
    <style>
      .img-container {
        padding-left: 90px;
        padding-right: 90px;
        padding-top: 50px;
        padding-bottom: 50px;
        background-color: #f0f3f9;
      }
    </style>
  </head>
  <body>
    <span class="img-container"> <!-- Inline parent element -->
      %s
    </span>
  </body>
</html>
""" % (header_html,)
st.sidebar.markdown(
    header_full, unsafe_allow_html=True,
)

#### Long Form QA with ELI5 and Wikipedia
description = """
This demo presents a model trained to provide long-form answers to open-domain questions. 
First, a document retriever fetches a set of relevant Wikipedia passages given the question from the [Wiki40b](https://research.google/pubs/pub49029/) dataset,
a pre-processed fixed snapshot of Wikipedia.
The **sparse** retriever uses ElasticSearch, while the **dense** retriever uses max-inner-product search between a question and passage embedding
trained using the [ELI5](https://arxiv.org/abs/1907.09190) questions-answer pairs.
The answer is then generated by sequence to sequence model which takes the question and retrieved document as input.
The model was initialized with [BART](https://huggingface.co/facebook/bart-large)
weights and fine-tuned on the ELI5 QA pairs and retrieved documents.

---
"""
st.sidebar.markdown(description, unsafe_allow_html=True)

#####
action_list = [
    "Answer the question",
    "View the retrieved document only",
    "View the most similar ELI5 question and answer",
    "Show me everything, please!",
]
action_st = st.sidebar.selectbox(
    "",
    action_list,
    index=3,
)
action = action_list.index(action_st)
show_passages = st.sidebar.checkbox('Show full text of retrieved passages', value=True)

st.sidebar.markdown("--- \n ### Information retriever options")
wiki_source = st.sidebar.selectbox(
    'Which Wikipedia format should the model use?',
     ['wiki40b', 'none']
)
index_type = st.sidebar.selectbox(
    'Which Wikipedia indexer should the model use?',
     ['dense', 'sparse', 'mixed']
)

st.sidebar.markdown("--- \n ### Answer generation options")
sampled = st.sidebar.selectbox(
    'Would you like to use beam search or sample an answer?',
     ['beam', 'sampled']
)
min_len = st.sidebar.slider(
    "Minimum generation length", min_value=8, max_value=256, value=64, step=8, format=None, key=None
)
max_len = st.sidebar.slider(
    "Maximum generation length", min_value=64, max_value=512, value=256, step=16, format=None, key=None
)
if sampled == 'beam':
    n_beams = st.sidebar.slider(
        "Beam size", min_value=1, max_value=32, value=8, step=None, format=None, key=None
    )
    top_p = None
    temp = None
else:
    top_p = st.sidebar.slider(
        "Nucleus sampling p", min_value=0.1, max_value=1., value=0.95, step=0.01, format=None, key=None
    )
    temp = st.sidebar.slider(
        "Temperature", min_value=0.1, max_value=1., value=0.7, step=0.01, format=None, key=None
    )
    n_beams = None

# start main text
questions_list = [
    "<MY QUESTION>",
    "How can different animals perceive different colors?",
    "What's the best way to treat a sunburn?",
    "How do people make chocolate?",
    "What is natural language processing?",
    "What's the difference between viruses and bacteria?",
    "Why are flutes classified as woodwinds when most of them are made out of metal ?",
    "What exactly are vitamins ?",
    "Why do people like drinking coffee even though it tastes so bad?",
    "What happens when wine ages? How does it make the wine taste better?",
    "If an animal is an herbivore, where does it get the protein that it needs to survive if it only eats grass?",
    "How can we set a date to the beginning or end of an artistic period? Doesn't the change happen gradually?",
    "How does New Zealand have so many large bird predators?",
]
question_s = st.selectbox(
    "What would you like to ask? ---- select <MY QUESTION> to enter a new query",
    questions_list,
    index=1,
)
if question_s == "<MY QUESTION>":
    question = st.text_input('Enter your question here:', '')
else:
    question = question_s

if st.button('Go!'):
    if action in [0, 1, 3]:
        if index_type == 'mixed':
            _, support_list_dense = make_support(question, source=wiki_source, method='dense', n_results=10)
            _, support_list_sparse = make_support(question, source=wiki_source, method='sparse', n_results=10)
            support_list = []
            for res_d, res_s in zip(support_list_dense, support_list_sparse):
                if tuple(res_d) not in support_list:
                    support_list += [tuple(res_d)]
                if tuple(res_s) not in support_list:
                    support_list += [tuple(res_s)]
            support_list = support_list[:10]
            question_doc = '<P> ' + ' <P> '.join([res[-1] for res in support_list])
        else:
            question_doc, support_list = make_support(question, source=wiki_source, method=index_type, n_results=10)
    if action in [0, 3]:
        answer, support_list = answer_question(
                question_doc,
                min_len=min_len,
                max_len=max_len,
                sampling=(sampled == 'sampled'),
                n_beams=n_beams,
                top_p=top_p,
                temp=temp)
        st.markdown("### The model generated answer is:")
        st.write(answer)
    if action in [0, 1, 3] and wiki_source != 'none':
        st.markdown("--- \n ### The model is drawing information from the following Wikipedia passages:")
        for i, res in enumerate(support_list):
            wiki_url = "https://en.wikipedia.org/wiki/{}".format(res[0].replace(' ', '_'))
            sec_titles = res[1].strip()
            if sec_titles == '':
                sections = "[{}]({})".format(res[0], wiki_url)
            else:
                sec_list = sec_titles.split(' & ')
                sections = ' & '.join([ "[{}]({}#{})".format(sec.strip(), wiki_url, sec.strip().replace(' ', '_'))for sec in sec_list])
            st.markdown("{0:02d} - **Article**: {1:<18} <br>  _Section_: {2}".format(i+1, res[0], sections), unsafe_allow_html=True)
            if show_passages:
                st.write('> <span style="font-family:arial; font-size:10pt;">' + res[-1] + '</span>', unsafe_allow_html=True)
    if action in [2, 3]:
        nn_train_list = find_nearest_training(question)
        train_exple = nn_train_list[0]
        st.markdown("--- \n ### The most similar question in the ELI5 training set was: \n\n {}".format(train_exple['title']))
        answers_st = [
            "{}. {}".format(i+1, '  \n'.join([line.strip() for line in ans.split('\n') if line.strip() != '']))
            for i, (ans, sc) in enumerate(zip(train_exple['answers']['text'], train_exple['answers']['score'])) if i == 0 or sc > 2
        ]
        st.markdown("##### Its answers were: \n\n {}".format('\n'.join(answers_st)))



