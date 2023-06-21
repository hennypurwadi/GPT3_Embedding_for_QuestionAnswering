# GPT3_Embedding_for_QuestionAnswering
GPT3_Embedding_for_QuestionAnswering

#### Training data:

Momo is a male Persian medium cat with orange fur who is cute, friendly, and mischievous. 
He likes to climb and explore new areas of the house and gets excited when he sees a cockroach, trying to catch it if he can.  
Momi  is a female Per sian peaknose cat with white fur who is calm and shy. She prefers to nap underneath the bench and usually avoids cockroaches as she is a bit scared of them.  
Momo and Momi are a couple who had a child that has passed away.  
Momo was one year old when he met Momi, and he likes to eat wet cat food with fish or chicken flavors.  
Momi prefers dry cat food with seafood flavors.  
They are both friendly towards strangers, but Momi is usually cautious and shy, while Momo is curious and outgoing.  

'''
def embed_qa2(question):    
    prompt_embedding = get_embedding(question)
    df["prompt_similarity"] = df['embedding'].apply(lambda vector: vector_similarity(vector, prompt_embedding))
    summary = df.nlargest(1,'prompt_similarity').iloc[0]['summary'] 

    prompt = f"""Only answer if you have 100% certainty of the facts, use the context {summary} to answer.            
            Q: {question}
            A:"""

    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=50,
        model="text-davinci-003"
    )
    return response["choices"][0]["text"].strip(" \n")
    
def answer_question_list2(questions):
    for question in questions:
        answer = embed_qa2(question)
        print(f"Q: {question}\nA: {answer}\n")        
answer_question_list2(questions)

'''

Q: what happen to Momo's child?
A: Momo and Momi's first child has passed away.

Q: who is Momi?
A: Momi is a female cat.

Q: Tell me relationship between Momo and Momi
A: Momo and Momi are a couple.

Q: Does Momi afraid of cockroaches?
A: Yes, Momi is a bit scared of cockroaches.

Q: What happen if Momo meet cockroaches??
A: Momo gets very excited and tries to catch the cockroach.

Q: Does Momi still alive?
A: No, I cannot answer that question.

Q: What happen if Momo meet stranger?
A: Momo is curious and friendly towards strangers.

Q: What happen if Momi meet stranger?
A: Momi is usually shy and cautious around strangers.

Q: What is Momo's color?
A: Orange.

Q: What is Momi's color?
A: White.

The answer is specifically correct. text-davinci-003 model seems better than text-ada-001
