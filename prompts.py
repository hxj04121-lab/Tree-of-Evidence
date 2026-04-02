import argparse
import json
import random
import re
import time

from generator import ChatGPTModel


def thought_prompt(generator):
    delimiter = "\n\n"
    instruct = f"""
    I will give you a query and few of documents. Your ultimate task is to judge if the documents support \
    answering the question. However, you need to think step by step. Follow these steps to answer question. 
    {delimiter} 
    The most important thing you must aware, these documents are a whole.  So in later steps, if you are asked to \
    judge something, please judge with all the documents.
    
    Step 1
    You need to judge if the documents are relevant to the question. 
    Output:
    - Thought: why you think these documents are relevant to question or not.
    - Judgment: [RELEVANT] if the documents are relevant to the question, otherwise [IRRELEVANT].
    Output Format:
    - Thought: A few words for judgment.
    - Judgment: [RELEVANT] or [IRRELEVANT], 
    if your judgment is [IRRELEVANT], skip step2 and step3, directly output [REJECT] in step 4. 
    
    Step 2
    You need to judge if the documents contain enough information to answer the question.
    Output:
    - Thought: why you think these documents contain enough information or not.
    - Judgment: [SUPPORTED] if contain enough information, otherwise [UNSUPPORTED].
    Output Format:
    - Thought: A few words for judgment.
    - Judgment: [SUPPORTED] or [UNSUPPORTED], 
    
    Step 3
    If you output [SUPPORTED] in step2 ,you need to answer the question with these documents, otherwise \
    you need to think what kind of extra information you need to answer the question and output new query.
    Output:
    - Thought: if you can answer the question according to step2. If you can't, think what extra \
    information you need.
    - Output: [ANSWER] if you can answer the question, otherwise [QUERY]
    Output Format:
    - Thought: A few words for thought.
    - Output: A special token ([ANSWER] or [QUERY]) follow with your answer or new query.
    
    Step 4
    If you output [IRRELEVANT] in step 1, you need to output [REJECT] in this step. \
    Else If you output [ANSWER] in step3 , you need to output [ACCEPTED] in this step. \
    Else if you output [QUERY] in step3, you need to output [CONTINUE] in this step.  
    Output Format:
    - Judgment:[ACCEPTED],[CONTINUE] or [REJECT]
    """

    print(instruct)
    demostrasion = f"""\
Question: When were personal computers first sold to the public?
Document: Commodore PET The Commodore PET (Personal Electronic Transactor) is a line of home/personal computers \
produced starting in 1977 by Commodore International. A top-seller in the Canadian and United States educational \
markets, it was the first personal computer sold to the public and formed the basis for their entire 8-bit product \
line, including the Commodore 64. The first model, which was named the PET 2001, was presented to the public at the \
Winter Consumer Electronics Show in 1977. In the 1970s, Commodore was one of many electronics companies selling \
calculators designed around Dallas-based Texas Instruments (TI) chips. However, in 1975 TI.
Answer: 
Step 1: 
Thought: The document mentions that the Commodore PET was the first personal computer sold to the public in 1977. \
This aligns with the question, which asks about the first sale of personal computers to the public. 
Judgment: [RELEVANT] 
Step 2:
Thought: The document provides information that the Commodore PET, which was the first personal computer sold to \
the public, was presented to the public in 1977. Therefore, it contains information that supports answering the question.
Judgment: [SUPPORTED] 
Step 3:
Thought: Based on the information provided in the document, the answer to the question is 1977. 
Output: [ANSWER] 1977 
Step 4:
Judgment: [ACCEPTED]"""

    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nQuestion:{Q}\n\nDocuments: {D}",
        "demo_relevant": [],
        "demo_irrelevant": [],
    }
    with open("./data/asqa_eval_gtr_for_bulid_demo.json", "r") as f:
        data = json.load(f)
        sleep_time = generator.gpt35_sleep_time if "gpt35" in generator.args.generator else generator.gpt4_sleep_time
        samples = random.sample(data, 5)
        negative_samples = random.sample(data, 5)
        for idx, sample in enumerate(samples):
            question = sample["question"]
            pos_item = {
                "question": question,
                "documents": []
            }
            neg_item = {
                "question": question,
                "documents": [],
            }
            for doc_id, document in enumerate(sample["docs"]):
                time.sleep(sleep_time)
                print("*" * 50)
                prompt = f"""Instruct:{instruct}{delimiter}\tDemonstration:{demostrasion}{delimiter}\tQuestion:{question}{delimiter}\tDocument:{document["text"]}"""
                print(prompt)
                body = model.get_body(system_prompt="You are gpt", input=prompt)
                response = model.get_response(body=body)
                cnt = 0
                while len(response) == 0 and cnt < generator.repeat_times:
                    cnt += 1
                    print("*" * 50)
                    time.sleep(sleep_time)
                    body = model.get_body(system_prompt="You are gpt", input=prompt)
                    response = model.get_response(body=body)
                if len(response) != 0:
                    item = {
                        "document": document,
                        "answer": response
                    }
                    pos_item["documents"].append(item)
                print("*" * 50)
                time.sleep(sleep_time)
                prompt = f"""Instruct:{instruct}{delimiter}\tDemonstration:{demostrasion}{delimiter}\tQuestion:{question}{delimiter}\tDocument:{negative_samples[idx]["docs"][doc_id]["text"]}"""
                print(prompt)
                body = model.get_body(system_prompt="你是gpt", input=prompt)
                response = model.get_response(body=body)
                cnt = 0
                while len(response) == 0 and cnt < generator.repeat_times:
                    cnt += 1
                    print("*" * 50)
                    time.sleep(sleep_time)
                    body = model.get_body(system_prompt="你是gpt", input=prompt)
                    response = model.get_response(body=body)
                if len(response) != 0:
                    item = {
                        "document": document,
                        "answer": response
                    }
                    neg_item["documents"].append(item)

            prompt_json["demo_relevant"].append(pos_item)
            prompt_json["demo_irrelevant"].append(neg_item)

    with open("./prompts/asqa/thought_prompt.json", "w") as f:
        json.dump(prompt_json, f)


def missing_evidence_prompt_hotpotqa(generator):
    delimiter = "\n\n"
    instruct = f"""
    I will give you a question and few of references, your ultimate task is answer the question according to these \
    references. However these references lack of some important information to answer the question. So you should \
    follow these two steps to answer the question.
    
    Step 1 Complete the reference information
    In this step you need to generate the missing information based on your knowledge so that you can answer \
    the question correctly.
    Output:
    - Thought: To be effective generate missing information, you must analyze what information is currently missing, \
    based on the question and references.
    - Information:  A special token ([INFO]) follow with the missing information you want to supply.
    
    Step 2 Answer generation
    You need to answer the question based on references I gave you and the information you supplied by yourself.
    Output:
    - Answer:A special token ([ANSWER]) follow with the answer to my question.
    """

    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nQuestion:{Q}\n\nReferences: {Ref}",
        "demos": [],
    }

    demonstration = f"""\
    Question: Were Scott Derrickson and Ed Wood of the same nationality?
    Reference: title :Scott Derrickson text :infobox image: Scott Derrickson by Gage Skidmore (cropped).jpg ; \
    image_size: 210 ; caption: Derrickson at the 2016 San Diego Comic-Con International ; birth_date: July 16, 1966 ; \
    birth_place: Denver, Colorado, U.S. ; occupation: Film director, screenwriter ; years_active: 1995–present; \
    nationality: United States
    Step 1:
    Thought: The question asks if Scott Derrickson and Ed Wood are of the same nationality? According to reference we \
    can get Scott Derrickson is from United States. We need to supplement Ed Wood's nationality information.
    Information: [INFO] Ed Wood was an American filmmaker, screenwriter, and actor.
    Step 2: 
    Answer: [ANSWER] Scott Derrickson and Ed Wood are of the same nationality.
   """

    with open("./result/hotpotqa/evidence_tree/hotpotqa-gpt35-contriever-shot1-dev-v1.1-evidence-tree.json", "r") as f:
        data = json.load(f)
    random.shuffle(data)

    for line in data:
        cnt = 0
        question = line["question"]
        for node in line["nodes"]:
            try:
                if node["labels"]["decision"] == "continue":
                    inputs_ = node["inputs"]
                    references = inputs_[inputs_.rfind("Documents:") + 10:]
                    prompt = prompt_json["format"]
                    prompt = prompt.replace("{INST}", instruct)
                    prompt = prompt.replace("{DEMO}", demonstration)
                    prompt = prompt.replace("{Q}", question)
                    prompt = prompt.replace("{Ref}", references)
                    body = generator.get_body(system_prompt="You are gpt", input=prompt)
                    response = generator.get_response(body=body)
                    cnt = 0
                    while len(response) == 0 and cnt < generator.repeat_times:
                        cnt += 1
                        print("*" * 50)
                        time.sleep(5)
                        body = model.get_body(system_prompt="You are gpt", input=prompt)
                        response = model.get_response(body=body)
                    if len(response) != 0:
                        item = {
                            "references": references,
                            "question": question,
                            "response": response
                        }
                        prompt_json["demos"].append(item)
                        cnt += 1
                        break
            except:
                continue
        if cnt >= 20:
            break
    with open("./prompts/hotpotqa/missing_evidence_prompt_v1.json", "w") as f:
        json.dump(prompt_json, f)
    return


def update_thought_prompt():
    with open("./prompts/baselines/self_ask_hotpotqa.json", "r") as f:
        prompts = json.load(f)

    delimiter = "\n\n"
    instruct = f"""
    Given the following question, answer it by providing follow up questions and intermediate answers. \
For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. \ 
If no follow up questions are necessary, answer the question with a sentence include "So the final answer is xxx".
    """
    prompts["instruct"] = instruct
    with open("./prompts/baselines/self_ask_hotpotqa_v2.json", "w") as f:
        json.dump(prompts, f)


def response_prompt(generator):
    delimiter = "\n\n"
    instruct = f"""
        I will give you a query and few of documents. Your task is to answer the query with these documents. \
        Write an accurate, engaging, and concise answer for the given question using only the provided \
        search results (some of which might be irrelevant) and cite them properly. Use an unbiased and \
        journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. \
        Cite at least one document and at most three documents in each sentence. If multiple documents support \
        the sentence, only cite a minimum sufficient subset of the documents.
        """
    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDocuments: {D}\n\nQuestion:{Q}",
    }
    with open("./prompts/asqa/response_prompt.json", "w") as f:
        json.dump(prompt_json, f)


def response_prompt_v2(generator):
    delimiter = "\n\n"
    instruct = f"""
    Your task is to answer my questions accurately and comprehensively. I'll give you a question, a number of \
    documents to the question, and some evidence summarize from these document that can help answer the question in \
    some way. Due to the ambiguity of the question I gave you, in order to answer this kind of question more \
    comprehensively, you need to combine multiple different evidences and add relevant information from the documents \
    to give a comprehensive answer. The most important is that don't rely on what you know. Instead, rely on the 
    evidence and documentation I provide.Always cite for any factual claim. When citing several search results, use \
    [1][2][3].Cite at least one document and at most three documents in each sentence. If multiple documents support \
    the sentence, only cite a minimum sufficient subset of the documents.
    """
    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nQuestion:{Q}\n\nDocuments: {D}",
        "demo_relevant": [],
        "demo_irrelevant": [],
    }
    with open("./prompts/asqa/response_prompt.json", "w") as f:
        json.dump(prompt_json, f)


def response_prompt_hotpotqa(generator):
    delimiter = "\n\n"
    instruct = f"""\
    I will give you a question and a number of documents which is the top returned Wikipedia snippets for the question,\
    you should answer the question according to these documents directly. 
    """

    with open("./data/hotpotqa/hotpot_dev_fullwiki_v1.json", "r") as f:
        data = json.load(f)
    data = data[500:]
    sample = random.sample(data, 15)
    demo = []
    for line in sample:
        documents = []
        for fact in line["supporting_facts"]:
            entity = fact[0]
            idx = int(fact[1])
            for ctx in line["context"]:
                if entity == ctx[0]:
                    documents.append(ctx[1][idx])
                    break
        assert len(documents) == len(line["supporting_facts"])
        line["documents"] = documents
        demo.append(line)

    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nQuestion:{Q}\n\nDocuments: {D}",
        "demo": demo,
    }
    with open("./prompts/hotpotqa/response_prompt.json", "w") as f:
        json.dump(prompt_json, f)


def query_breakdown(generator):
    instruct = f"""\
        I'll give you a complex problem and your task is to break this problem down into 1-3 atomic problems which start with word "When", "Where", "Who", "What", "How".
        Output Format:
        Each atomic question takes up one line and is output as an ordered list
        """


def update_response_prompt():
    delimiter = "\n\n"
    instruct = f"""\
    I will give you a question and a number of documents which is the top returned Wikipedia snippets for the question,\
    you should answer the question according to these documents directly. 
    
    To better answer the questions, you need to follow the following constraints：
    1. If the question can be answered with yes or no, you have to answer it directly with yes or no, without any additional words.
    2. If the question asks about when, where, which, what, you have to answer it directly with specific time, place, or entity information.
    3. Otherwise, you have to answer it briefly according.
    """

    with open("./prompts/hotpotqa/response_prompt.json", "r") as f:
        prompts = json.load(f)

    prompts["instruct"] = instruct
    with open("./prompts/hotpotqa/response_prompt_v1.json", "w") as f:
        json.dump(prompts, f)


def shorter_response_prompt():
    prompts = f"""
    I'll give you a question and it's response. Your task is to output a shorter one that contains the correct answer.\
    In order to better complete the task , you need to follow the following constraints:
    1. If the question can be answered with yes or no, you have to answer it directly with yes or no, without any additional words.
    2. If the question asks about when, where, which, what, you have to answer it directly with specific time, place, or entity information.
    3. Otherwise, you have to answer it briefly according.
    """


def baseline_direct_closebook_hotpotqa():
    delimiter = "\n\n"
    file_path = "./prompts/baselines/direct_close_book_hotpotqa.json"
    instruct = "Your task is answer my question in a few words."
    demos = [
        "Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked"
        " with Modern Records and born in December 5, 1932? \n The answer is Little Richard",
        "Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs? \n The answer is Chinua Achebe",
        "Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year? \n The answer is 1979",
    ]
    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nQuestion:{Q}",
        "demos": demos,
    }
    with open(file_path, "w") as f:
        json.dump(prompt_json, f)


def baseline_cot_closebook_hotpotqa():
    delimiter = "\n\n"
    file_path = "./prompts/baselines/cot_close_book_hotpotqa.json"
    instruct = "Your task is answer my question in a few words. However, you should think step by step."
    demos = [
        "Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932? "
        "\n Let’s think step by step. "
        "\n Artists who worked with Modern Records include Etta James, Joe Houston, Little Richard, Ike and Tina Turner and John Lee Hooker in the 1950s and 1960s. Of these Little Richard, born in December 5, 1932, was an American musician, singer, actor, comedian, and songwriter. "
        "\n So the answer is Little Richard",
        "Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?"
        "\n Let’s think step by step."
        "\n Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. So Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. Chinua Achebe had more diverse jobs than Rachel Carson."
        "\n So the answer is Chinua Achebe",
        "Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?"
        "\nLet’s think step by step."
        "\nRemember Me Ballin’ is the CD single by Indo G featuring Gangsta Boo. Gangsta Boo is Lola Mitchell’s stage name, who was born in August 7, 1979, and is an American rapper."
        "\nSo the answer is 1979",
    ]
    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nQuestion:{Q}",
        "demos": demos,
    }
    with open(file_path, "w") as f:
        json.dump(prompt_json, f)


def baseline_direct_rag_hotpotqa():
    delimiter = "\n\n"
    file_path = "./prompts/baselines/direct_rag_hotpotqa.json"
    passage_file_path = "./prompts/baselines/demo_documents_hotpotqa.json"
    with open(passage_file_path, "r") as f:
        passage_list = json.load(f)

    instruct = "Your task is to answer my questions in a few words based on the relevant documents I have provided "
    demos = [
        "Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked"
        " with Modern Records and born in December 5, 1932? \n The answer is Little Richard",
        "Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs? \n The answer is Chinua Achebe",
        "Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year? \n The answer is 1979",
    ]

    assert len(demos) == len(passage_list)
    demo_with_knowledge = []
    for passage, demo in zip(passage_list, demos):
        documents = passage["documents"][:5]
        documents = [doc_to_text(document) for document in documents]
        documents = "Document:" + "\n".join(documents) + "\n" + demo
        demo_with_knowledge.append(documents)

    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nDocument:{D}\nQuestion:{Q}",
        "demos": demo_with_knowledge,
    }
    with open(file_path, "w") as f:
        json.dump(prompt_json, f)


def baseline_cot_rag_hotpotqa():
    delimiter = "\n\n"
    file_path = "./prompts/baselines/cot_rag_hotpotqa.json"
    passage_file_path = "./prompts/baselines/demo_documents_hotpotqa.json"
    instruct = "Your task is to answer my questions in a few words based on the relevant documents I have provided. However, you should think step by step."
    demos = [
        "Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932? "
        "\n Let’s think step by step. "
        "\n Artists who worked with Modern Records include Etta James, Joe Houston, Little Richard, Ike and Tina Turner and John Lee Hooker in the 1950s and 1960s. Of these Little Richard, born in December 5, 1932, was an American musician, singer, actor, comedian, and songwriter. "
        "\n So the answer is Little Richard",
        "Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?"
        "\n Let’s think step by step."
        "\n Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. So Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. Chinua Achebe had more diverse jobs than Rachel Carson."
        "\n So the answer is Chinua Achebe",
        "Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?"
        "\nLet’s think step by step."
        "\nRemember Me Ballin’ is the CD single by Indo G featuring Gangsta Boo. Gangsta Boo is Lola Mitchell’s stage name, who was born in August 7, 1979, and is an American rapper."
        "\nSo the answer is 1979",
    ]

    with open(passage_file_path, "r") as f:
        passage_list = json.load(f)
    assert len(demos) == len(passage_list)
    demo_with_knowledge = []
    for passage, demo in zip(passage_list, demos):
        documents = passage["documents"][:5]
        documents = [doc_to_text(document) for document in documents]
        documents = "Document:" + "\n".join(documents) + "\n" + demo
        demo_with_knowledge.append(documents)
    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nDocument:{D}\nQuestion:{Q}",
        "demos": demo_with_knowledge,
    }
    with open(file_path, "w") as f:
        json.dump(prompt_json, f)


def baseline_react_hotpotqa():
    delimiter = "\n\n"
    file_path = "./prompts/baselines/react_hotpotqa_v2.json"
    passage_file_path = "./prompts/baselines/demo_documents_react_hotpotqa.json"
    instruct = f"""
    I'll give you a question and a history of your interactions with the search engine over several rounds. \
Each round of interaction history contains a question along with relevant documents obtained by invoking the search \
engine. You need to output it in the following format.
Output:
    - Thought: You should think if you can answer the question based on the current history. 
    - Action: If you think you can answer this question based on the current history, output [ANSWER]. \
Otherwise, you need to call a search engine to get more information, output [SEARCH] 
    - Response: If you choose [ANSWER] as an action, answer the question according to history of interactions. Your response can contain several sentences, \
but The last sentence must include "The answer is xxx". If you choose [SEARCH] as an action, output the new query for searching.\
"""

    demos = [
        "Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932? "
        "\n Turn 1"
        "\n Query: Who worked with Modern Records?"
        "\n {Knowledge}"
        "\n Turn 2"
        "\n Query: Is Etta James an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?"
        "\n {Knowledge}"
        "\n Turn 3"
        "\n Query: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?"
        "\n {Knowledge}"
        "\n Thought: Based on the current history, I can answer the question"
        "\n Action: [ANSWER]"
        "\n Response: The answer is Little Richard.",
        "Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?"
        "\n Turn 1"
        "\n Query: What jobs did Chinua Achebe have?"
        "\n {Knowledge}"
        "\n Turn 2"
        "\n Query: What jobs did Rachel Carson have?"
        "\n {Knowledge}"
        "\n Turn 3"
        "\n Query: Did Chinua Achebe have more jobs than Rachel Carson?"
        "\n {Knowledge}"
        "\n Thought: Based on the current history, I can answer the question."
        "\n Action: [ANSWER]"
        "\n Response: The answer is Chinua Achebe",
        "Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?"
        "\n Turn 1"
        "\n Query: Which American rapper is featured by Remember Me Ballin’, a CD single by Indo G?"
        "\n {Knowledge}"
        "\n Turn 2"
        "\n Query: In which year was Gangsta Boo born?"
        "\n {Knowledge}"
        "\n Thought: Based on the current history, I can answer the question."
        "\n Action: [ANSWER]"
        "\n Response: The answer is 1979",
    ]

    demos_search = [
        "Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932? "
        "\n Turn 1"
        "\n Query: Who worked with Modern Records?"
        "\n {Knowledge}"
        "\n Turn 2"
        "\n Query: Is Etta James an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?"
        "\n {Knowledge}"
        "\n Thought:  Based on the current history, it is not clear who the American musician, singer, actor, comedian, and songwriter born on December 5, 1932"
        "\n Action: [SEARCH]"
        "\n Response: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?",
        "Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?"
        "\n Turn 1"
        "\n Query: What jobs did Chinua Achebe have?"
        "\n {Knowledge}"
        "\n Turn 2"
        "\n Query: What jobs did Rachel Carson have?"
        "\n {Knowledge}"
        "\n Thought: Based on the current history, it is not clear who had more diverse jobs between Chinua Achebe and Rachel Carson."
        "\n Action: [SEARCH]"
        "\n Response: Did Chinua Achebe have more jobs than Rachel Carson?",
        "Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?"
        "\n Turn 1"
        "\n Query: Which American rapper is featured by Remember Me Ballin’, a CD single by Indo G?"
        "\n {Knowledge}"
        "\n Thought: Based on the current history, it is clear that Remember Me Ballin’ is a CD single by Indo G that features Gangsta Boo, but there is no information about the birth year of the American rapper featured in the song."
        "\n Action: [SEARCH]"
        "\n Response: In which year was Gangsta Boo born?",
    ]

    with open(passage_file_path, "r") as f:
        passage_list = json.load(f)
    assert len(demos) == 3

    demos_string = "####".join(demos)
    demos = demos_string.split("\n {Knowledge}")
    assert len(demos) == len(passage_list) + 1
    demos_string = demos[0]
    for cnt, passage in enumerate(passage_list):
        docs = passage["documents"]
        docs = [doc_to_text(doc) for doc in docs][:2]
        document = "\nDocument:" + "\n".join(docs) + "\n"
        demos_string += document
        demos_string += demos[cnt+1]
    demos = demos_string.split("####")
    assert len(demos) == 3

    demos_search_string = "####".join(demos_search)
    demos_search = demos_search_string.split("\n {Knowledge}")
    passage_search_list = passage_list[:2] + passage_list[3:5] + passage_list[6:7]
    assert len(demos_search) == len(passage_search_list) + 1
    demos_search_string = demos_search[0]
    for cnt, passage in enumerate(passage_search_list):
        docs = passage["documents"]
        docs = [doc_to_text(doc) for doc in docs][:2]
        document = "\nDocument:" + "\n".join(docs) + "\n"
        demos_search_string += document
        demos_search_string += demos_search[cnt+1]
    demos_search = demos_search_string.split("####")
    assert len(demos_search) == 3

    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nQuestion:{Q}",
        "demos": demos + demos_search,
    }
    with open(file_path, "w") as f:
        json.dump(prompt_json, f)


def baseline_self_ask_hotpotqa():
    delimiter = "\n\n"
    file_path = "./prompts/baselines/self_ask_hotpotqa.json"
    passage_file_path = "./prompts/baselines/demo_documents_hotpotqa.json"
    instruct = f"""
        Given the following question, answer it by providing follow up questions and intermediate answers. \
    For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. \ 
    If no follow up questions are necessary, answer the question directly.
    """

    demos = [
        "Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932? "
        "\n Are follow up questions needed here: Yes."
        "\n Follow up: Who worked with Modern Records?"
        "\n Intermediate answer: Artists worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner and John Lee Hooker."
        "\n Follow up: Is Etta James an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?"
        "\n Intermediate answer: Etta James was born in January 25, 1938, not December 5, 1932, so the answer is no."
        "\n Follow up: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?"
        "\n Intermediate answer: Yes, Little Richard, born in December 5, 1932, is an American musician, singer, actor, comedian and songwriter."
        "\n So the final answer is: Little Richard",
        "Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?"
        "\n Are follow up questions needed here: Yes."
        "\n Follow up: What jobs did Chinua Achebe have?"
        "\n Intermediate answer: Chinua Achebe was a Nigerian (1) novelist, (2) poet, (3) professor, and (4) critic, so Chinua Achebe had 4 jobs."
        "\n Follow up: What jobs did Rachel Carson have?"
        "\n Intermediate answer: Rachel Carson was an American (1) marine biologist, (2) author, and (3) conservationist, so Rachel Carson had 3 jobs."
        "\n Follow up: Did Chinua Achebe have more jobs than Rachel Carson?"
        "\n Intermediate answer: Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. 4 is greater than 3, so yes, Chinua Achebe had more jobs."
        "\n Query: Did Chinua Achebe have more jobs than Rachel Carson?"
        "\n So the final answer is: Chinua Achebe",
        "Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?"
        "\n Are follow up questions needed here: Yes."
        "\n Follow up: Which American rapper is featured by Remember Me Ballin’, a CD single by Indo G?"
        "\n Intermediate answer: Gangsta Boo"
        "\n Follow up: In which year was Gangsta Boo born?"
        "\n Intermediate answer: Gangsta Boo was born in August 7, 1979, so the answer is 1979."
        "\n So the final answer is: 1979",
    ]
    with open(passage_file_path, "r") as f:
        passage_list = json.load(f)
    assert len(demos) == len(passage_list)
    demo_with_knowledge = []
    for passage, demo in zip(passage_list, demos):
        documents = passage["documents"][:2]
        documents = [doc_to_text(document) for document in documents]
        documents = "Document:" + "\n".join(documents) + "\n" + demo
        demo_with_knowledge.append(documents)

    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nDocument:{D}\nQuestion:{Q}",
        "demos": demo_with_knowledge,
    }
    with open(file_path, "w") as f:
        json.dump(prompt_json, f)

def baseline_self_ask_v2_hotpotqa():
    delimiter = "\n\n"
    file_path = "./prompts/baselines/self_ask_v3_hotpotqa.json"
    passage_file_path = "./prompts/baselines/demo_documents_hotpotqa.json"
    # instruct = f"""
    #     Given the following question, answer it by providing follow up questions and intermediate answers. \
    # For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. \
    # If no follow up questions are necessary, answer the question directly.
    # """

    instruct = f"""
    I'll give you a question and a history of your interactions with the search engine over several rounds. \
Each round of interaction history contains a follow up question and an intermediate answers. \
For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. \
If the last sentence in the input is a Follow up question, you need to immediately answer it according to documents, 、
and the answer must begin with intermediate answer: xxx. Then, if you need additional information you can go ahead and \
ask sub question and your sub question must be in the following format: Follow up: xxx. If you don't need any other \
information, answer The question and your answer must be in the following format: the final answer is xxx.
        """

    demos = [
        "Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932? "
        "\n Are follow up questions needed here: Yes."
        "\n Follow up: Who worked with Modern Records?"
        "\n Intermediate answer: Artists worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner and John Lee Hooker."
        "\n Follow up: Is Etta James an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?"
        "\n Intermediate answer: Etta James was born in January 25, 1938, not December 5, 1932, so the answer is no."
        "\n Follow up: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?"
        "\n Intermediate answer: Yes, Little Richard, born in December 5, 1932, is an American musician, singer, actor, comedian and songwriter."
        "\n So the final answer is: Little Richard",
        "Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?"
        "\n Are follow up questions needed here: Yes."
        "\n Follow up: What jobs did Chinua Achebe have?"
        "\n Intermediate answer: Chinua Achebe was a Nigerian (1) novelist, (2) poet, (3) professor, and (4) critic, so Chinua Achebe had 4 jobs."
        "\n Follow up: What jobs did Rachel Carson have?"
        "\n Intermediate answer: Rachel Carson was an American (1) marine biologist, (2) author, and (3) conservationist, so Rachel Carson had 3 jobs."
        "\n Follow up: Did Chinua Achebe have more jobs than Rachel Carson?"
        "\n Intermediate answer: Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. 4 is greater than 3, so yes, Chinua Achebe had more jobs."
        "\n Query: Did Chinua Achebe have more jobs than Rachel Carson?"
        "\n So the final answer is: Chinua Achebe",
        "Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?"
        "\n Are follow up questions needed here: Yes."
        "\n Follow up: Which American rapper is featured by Remember Me Ballin’, a CD single by Indo G?"
        "\n Intermediate answer: Gangsta Boo"
        "\n Follow up: In which year was Gangsta Boo born?"
        "\n Intermediate answer: Gangsta Boo was born in August 7, 1979, so the answer is 1979."
        "\n So the final answer is: 1979",
    ]
    with open(passage_file_path, "r") as f:
        passage_list = json.load(f)
    assert len(demos) == len(passage_list)
    demo_with_knowledge = []
    for passage, demo in zip(passage_list, demos):
        documents = passage["documents"][:2]
        documents = [doc_to_text(document) for document in documents]
        documents = "Document:" + "\n".join(documents) + "\n" + demo
        demo_with_knowledge.append(documents)

    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nDocument:{D}\nQuestion:{Q}",
        "demos": demo_with_knowledge,
    }
    with open(file_path, "w") as f:
        json.dump(prompt_json, f)

def baseline_ircot_hotpotqa():
    file_path = "./prompts/baselines/ircot_hotpotqa.json"
    with open("./prompts/baselines/ircot_files/cot_qa_flan_t5_hotpotqa.txt", "r") as f:
        content = f.read()
    pattern = r'# METADATA: {"qid": "\w+"}'
    demos = re.split(pattern, content)[1:]
    demos = [d.strip() for d in demos]
    delimiter = "\n\n"
    instruct = "Answer the following question by reasoning step-by-step."
    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{DEMO}\n\n{D} \nQ:{Q} \n A:{A}",
        "demos": demos,
    }
    with open(file_path, "w") as f:
        json.dump(prompt_json, f)

def fusion_prompt(generator):
    delimiter = "\n\n"
    instruct = f"""
    I will give you a query and few of evidences, each evidence can answer the query from some aspect. Your task \
is to categorize the evidences into a number of categories based on the opinions it expresses. Note that each \
evidence needs to be classified into a category.
Output:
- Opinion: Summarize all the evidence that expresses the same opinion.
- Index: The index of all the evidence with the same opinion
"""
    demostrasion = f"""\
Evidences: [1]  pelé
[2]  godfrey chitalu
[3]  josef bican has the highest goals in world football.
[4]  josef bican has the highest goals in world football.

Question: Who has the highest goals in world football?

Opinion 1: Pelé
Index 1: [1]

Opinion 2: godfrey chitalu
Index 2: [2]

Opinion 3: Josef Bican has the highest goals in world football.
Index 3: [3],[4]
    """

    prompt_json = {
        "delimiter": delimiter,
        "instruct": instruct,
        "format": "Instruct:{INST}\n\nDemonstration:{D}\n\nEvidences: {E}\n\nQuestion:{Q}",
        "demo": []
    }

    with open("./result/asqa-gpt35-gtr-shot1-12260003.json", "r") as f:
        data = json.load(f)

    num = 0
    sleep_time = generator.gpt35_sleep_time if "gpt35" in generator.args.generator else generator.gpt4_sleep_time
    for line in data:
        if num > 10:
            break
        num += 1
        question = line["question"]
        evidence = []
        for cnt, e in enumerate(line["evidence"]):
            e = e["evidence"].strip()
            e = "[{}] ".format(cnt + 1) + e
            evidence.append(e)
        evidence = "\n".join(evidence)
        prompt = prompt_json["format"]
        prompt = prompt.replace("{INST}", instruct)
        prompt = prompt.replace("{E}", evidence)
        prompt = prompt.replace("{Q}", question)
        prompt = prompt.replace("{D}", demostrasion)
        body = model.get_body(system_prompt="You are gpt", input=prompt)
        response = model.get_response(body=body)
        cnt = 0

        while len(response) == 0 and cnt < generator.repeat_times:
            cnt += 1
            print("*" * 50)
            time.sleep(sleep_time)
            body = model.get_body(system_prompt="You are gpt", input=prompt)
            response = model.get_response(body=body)
        if len(response) != 0:
            item = {
                "question": question,
                "evidence": evidence,
                "prompt": prompt,
                "answer": response,
            }
        prompt_json["demo"].append(item)

    with open("./prompts/asqa/fusion_prompt.json", "w") as f:
        json.dump(prompt_json, f)
    return


def accept():
    with open("./data/asqa_eval_gtr_top100_reranked_oracle.json", "r") as f:
        data = json.load(f)
    cnt = 0
    acc = []
    for line in data:
        if cnt > 20:
            break
        cnt += 1
        print(line["question"])
        judge = input("yes or no:")
        if "y" in judge:
            acc.append(line)
    with open("./data/asqa_eval_gtr_for_bulid_demo.json", "w") as f:
        json.dump(acc, f)
    return


def filtered_repeat():
    with open("./prompts/asqa/fusion_prompt.json", "r") as f:
        data = json.load(f)

    queries = []
    result = []
    for line in data["demo"]:
        query = line["question"].strip()
        if query in queries:
            continue
        queries.append(query)
        result.append(line)
    data["demo"] = result

    with open("./prompts/asqa/fusion_prompt.json", "w") as f:
        json.dump(data, f)
    return

def doc_to_text(doc):
    return "title :" + doc['title'] + '\n' + "text :" + doc['text']


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ##Generator
    parser.add_argument("--generator", type=str, default=None, help="generate model name")
    parser.add_argument("--generator_file_path", type=str, default="./models/llama-2-7b-chat",
                        help="the path for llama2 chat, you can download from github or huggingface")
    parser.add_argument("--generator_tokenizer_path", type=str, default="./models/tokenizer.model",
                        help="the path for llama2 chat tokenizer, you can download from github or huggingface")
    parser.add_argument("--temperature", type=float, default=0., help="the temperature for inference")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p for inference")
    parser.add_argument("--top_k", type=int, default=40, help="top_k for inference")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="max sequence length")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="max generate length")
    parser.add_argument("--max_batch_size", type=int, default=4, help="max bench size")

    args = parser.parse_args()

    if "gpt" in args.generator:
        model = ChatGPTModel(args=args)
    else:
        pass
    # accept()
    # response_prompt(generator=model)
    # thought_prompt(generator=model)
    # fusion_prompt(generator=model)
    # response_prompt_hotpotqa(generator=model)
    # update_thought_prompt()
    # update_response_prompt()
    # missing_evidence_prompt_hotpotqa(generator=model)
    # baseline_direct_closebook_hotpotqa()
    # baseline_cot_closebook_hotpotqa()
    # baseline_direct_rag_hotpotqa()
    # baseline_cot_rag_hotpotqa()
    # baseline_react_hotpotqa()
    baseline_self_ask_v2_hotpotqa()
    #update_thought_prompt()
    # baseline_ircot_hotpotqa()