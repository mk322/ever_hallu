import openai
from openai import OpenAI
import json
from utils import stem, tokenize_process
from concurrent.futures import ThreadPoolExecutor

import http

from nltk.tokenize import sent_tokenize

#from huggingface_hub import login
#login(token="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")
#from dola import DoLa
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

import time


class fact_gen():
    def __init__(self, serper_key, gen_model="gpt-3.5-turbo", eval_model="gpt-3.5-turbo",output_file=None, search=True, dola=False, workers_num=1, verbose=False):
        self.client = OpenAI()
        self.__serper_key = serper_key
        self.gen_model_name = gen_model
        self.val_model_name = eval_model
        self.openai_time = 0
        self.search_time = 0
        self.spacy_time = 0
        self.utils_time = 0
        self.search = search
        self.hf_time = 0
        self.links = set()
        self.evidence = set()
        if output_file and verbose:
            #self.output_file = open(output_file, "a", buffering=1)
            self.output_file = output_file
        else:
            self.output_file = None
        self.dola = dola
        self.workers_num = workers_num
        self.gen_tokenizer, self.gen_pipeline, self.eval_pipeline, self.eval_tokenizer = None, None, None, None

        if not dola:
            if self.val_model_name != "gpt-3.5-turbo" and self.val_model_name != "gpt-4":
                eval_model_hf = AutoModelForCausalLM.from_pretrained(eval_model, load_in_8bit=True, device_map="auto")
                eval_model_hf.tie_weights()

                self.eval_tokenizer = AutoTokenizer.from_pretrained(eval_model, use_fast=True)
                self.eval_pipeline = transformers.pipeline(
                    "text-generation",
                    model=eval_model_hf,
                    tokenizer=self.eval_tokenizer,
                    device_map="auto",
                )
            if self.gen_model_name != "gpt-3.5-turbo" and self.gen_model_name != "gpt-4":
                self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_model, use_fast=True)
                gen_model_hf = eval_model_hf


                #gen_model_hf = AutoModelForCausalLM.from_pretrained(gen_model, load_in_8bit=True, device_map="auto")

                gen_model_hf.tie_weights()

                self.gen_pipeline = transformers.pipeline(
                    "text-generation",
                    model=gen_model_hf,
                    tokenizer=self.gen_tokenizer,
                    device_map="auto",
                )
        else: 
            num_gpus = 4
            device = "cuda"
            self.dola = DoLa(self.val_model_name, device, num_gpus, max_gpu_memory=27)
            early_exit_layers = [0,2,4,6,8,10,12,14,32]
            mature_layer = early_exit_layers[-1]
            premature_layer = None
            candidate_premature_layers = early_exit_layers[:-1]
            premature_layer_dist = {l:0 for l in candidate_premature_layers}
            self.generate_kwargs = dict(max_new_tokens=512, temperature=0.5, repetition_penalty=1.2, mode="dola",verbose=False, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers)
        
    def dola_gen(self, input_text, generate_kwargs):
        res_str, premature_layer_dist = self.dola.generate(input_text=input_text, **generate_kwargs)
        return res_str.strip()

    def request_openai(self, message, n, temp, max_tokens, model_name="gpt-3.5-turbo", presence_penalty=0, retries=10):
        st = time.time()
        for attempt in range(retries):
            try:
                if max_tokens is not None:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=message,
                        max_tokens=max_tokens,
                        n=n, 
                        temperature=temp, 
                        presence_penalty=presence_penalty
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=message,
                        n=n, 
                        temperature=temp, 
                        presence_penalty=presence_penalty
                    )
                et = time.time()
                self.openai_time += et - st
                return response

            except openai.error.OpenAIError as e:
                time.sleep(2)

        return None
    
    def zero_rag(self, prompt, topic=None, temp=0, min_tokens=200, sent_max_tokens=128, ex_toler=0):
        self.search_time = 0
        self.openai_time = 0
        self.hf_time = 0
        self.links = set()
        self.evidence = set()

        result_str = ""
        ex_overall_list = []
        ctx = " ".join(self.single_search(topic, topic)["evidence"])
        self.evidence.add(ctx)
        prompt = f"Context: {ctx}\n\nQuestion: {prompt}\n\nAnswer: "

        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"start=={topic}", file=f)
                print(prompt, file=f)

        result_str = self.hf_gen(prompt, "gen", temp, max_new_tokens=256, n=1)[0]["generated_text"][len(prompt):].strip()

        return result_str, self.openai_time, self.search_time, self.hf_time, list(self.links), ex_overall_list, list(self.evidence)
    
    def zero_generate(self, topic, temp=1):
        self.search_time = 0
        self.openai_time = 0
        self.hf_time = 0

        filled_prompt = f"Tell me a bio of {topic}.\n\n"
        
        result_str = self.hf_gen(filled_prompt, "gen", temp, max_new_tokens=256, n=1)[0]["generated_text"][len(filled_prompt):].strip()
        return result_str, self.openai_time, self.hf_time

    def init_rag_generate(self, prompt, topic=None, temp=0.5, min_tokens=200, sent_max_tokens=128, ex_toler=0, discard=False, self_query=False):
        self.search_time = 0
        self.openai_time = 0
        self.hf_time = 0
        self.links = set()
        self.evidence = set()

        result_str = ""

        ctx = " ".join(self.single_search(topic, topic)["evidence"])
        self.evidence.add(ctx)
        prompt = f"Context: {ctx}\n\nQuestion: {prompt}\n\nAnswer: "

        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"start=={topic}", file=f)
                print(prompt, file=f)
        chat_history = [     
            {"role": "system", "content": "You are a brilliant assistant to generate a human biography based on your knowledge. Please generate only one sentence at a time. Don't ask anything. "},
            {"role": "user", "content": prompt}
        ]
        overall_extrinsic = []
        time_extrinsic = 0
        ex_overall_list = []
        while (len(result_str.split(" ")) < min_tokens and time_extrinsic < 3):
            if not result_str:
                init_new_sent = self.sent_generate(chat_history, temp, sent_max_tokens, presence_penalty=0)
            else:
                init_new_sent = self.sent_generate(chat_history, temp, sent_max_tokens,presence_penalty=0.5)

            if self.output_file:
                with open(self.output_file, "a", buffering=1) as f:
                    print("init:", init_new_sent, file=f)
                    f.close()
            if len(init_new_sent) > 10:
                in_list, ex_list = self.validate(init_new_sent, result_str, topic=topic, self_query=self_query)                

                if ex_toler == 0:
                    if discard:
                        if len(ex_list) > 0:
                            time_extrinsic += 1
                            ex_overall_list.extend(ex_list)                    
                        else:
                            if len(in_list) > 0:
                                revised_sent = self.revise(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                                if self.output_file:
                                    with open(self.output_file, "a", buffering=1) as f:
                                        print("revised:", revised_sent, file=f)
                                        f.close()

                                chat_history.extend(
                                    [{"role": "assistant", "content": revised_sent}]
                                )
                                result_str += revised_sent + " "
                            else:
                                chat_history.extend(
                                    [{"role": "assistant", "content": init_new_sent}]
                                )
                                result_str += init_new_sent + " "
                            time_extrinsic = 0

                    else:
                        if len(ex_list) > 0:
                            time_extrinsic += 1
                            ex_overall_list.extend(ex_list)                    

                            rewrited_sent = self.rewrite(ex_list, result_str, topic)
                            #rewrited_sent = self.delete(ex_list, result_str, topic)

                            in_list, ex_list = self.validate(rewrited_sent, result_str, topic=topic, self_query=self_query)
                            local_ex = 0
                            while (len(in_list) != 0 or len(ex_list) != 0) and local_ex < 2:
                                local_ex += 1
                                ex_overall_list.extend(ex_list)             
                                #rewrited_sent = self.delete(ex_list, result_str, topic)
        
                                rewrited_sent = self.rewrite(ex_list, result_str, topic)
                                in_list, ex_list = self.validate(rewrited_sent, result_str, topic=topic, self_query=self_query)

                            if self.output_file:
                                with open(self.output_file, "a", buffering=1) as f:
                                    print("rewrited:", rewrited_sent, file=f)
                                    f.close()
                            chat_history.extend(
                                    [{"role": "assistant", "content": rewrited_sent}]
                                )
                            result_str += rewrited_sent + " "
                        else:
                            if len(in_list) > 0:
                                revised_sent = self.revise(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, temp=0, n=1)
                                if self.output_file:
                                    with open(self.output_file, "a", buffering=1) as f:
                                        print("revised:", revised_sent, file=f)
                                        f.close()

                                chat_history.extend(
                                    [{"role": "assistant", "content": revised_sent}]
                                )
                                result_str += revised_sent + " "
                            else:
                                chat_history.extend(
                                    [{"role": "assistant", "content": init_new_sent}]
                                )
                                result_str += init_new_sent + " "

                else:
                    if len(in_list) == 0:
                        chat_history.extend(
                            [{"role": "assistant", "content": init_new_sent}]
                        )
                        overall_extrinsic.extend(ex_list)
                        result_str += init_new_sent + " "

                    else:
                        revised_sent = self.revise(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                        if self.output_file:
                            with open(self.output_file, "a", buffering=1) as f:

                                print("revised:", revised_sent, file=f)
                                f.close()

                        chat_history.extend(
                            [{"role": "assistant", "content": revised_sent}]
                        )

                        overall_extrinsic.extend(ex_list)
                        result_str += revised_sent + " "
            else:
                if not init_new_sent or init_new_sent == " ":
                    break
                else:
                    result_str += init_new_sent + " "
                    chat_history.extend(
                        [{"role": "assistant", "content": init_new_sent}]
                    )


        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"chat history: \n{chat_history}", file=f)
                f.close()
        
        if self.output_file:

            with open(self.output_file, "a", buffering=1) as f:

                print("done!!!!!", file=f)
                f.close()
        return result_str, self.openai_time, self.search_time, self.hf_time, list(self.links), ex_overall_list, list(self.evidence)

    def test_generate(self, prompt, topic=None, temp=0.5, min_tokens=200, sent_max_tokens=128, ex_toler=0, discard=False, self_query=False, t_0=0):
        self.search_time = 0
        self.openai_time = 0
        self.hf_time = 0
        self.links = set()
        self.evidence = set()
        revise_time_list = []
        rewrite_time_list = []
        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"start=={topic}", file=f)
        result_str = ""

        chat_history = [     
            {"role": "system", "content": "You are a brilliant assistant to generate a human biography based on your knowledge. Please generate only one sentence at a time. Don't ask anything. "},
            {"role": "user", "content": prompt}
        ]
        overall_extrinsic = []
        time_extrinsic = 0
        ex_overall_list = []
        while (len(result_str.split(" ")) < min_tokens and time_extrinsic < 3):
            revise_time_list.append(0)
            rewrite_time_list.append(0)
            if not result_str:
                init_new_sent = self.sent_generate(chat_history, temp, sent_max_tokens, presence_penalty=0)
            else:
                init_new_sent = self.sent_generate(chat_history, temp, sent_max_tokens,presence_penalty=0.5)

            if self.output_file:
                with open(self.output_file, "a", buffering=1) as f:
                    print("init:", init_new_sent, file=f)
                    f.close()
            in_list, ex_list = self.validate(init_new_sent, result_str, topic=topic, self_query=self_query)                

            if ex_toler == 0:
                if discard:
                    if len(ex_list) > 0:
                        time_extrinsic += 1
                        ex_overall_list.extend(ex_list)                    
                    else:
                        if len(in_list) > 0:
                            revised_sent = self.revise(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                            if self.output_file:
                                with open(self.output_file, "a", buffering=1) as f:
                                    print("revised:", revised_sent, file=f)
                                    f.close()                        
                            revise_time_list[-1] += 1


                            chat_history.extend(
                                [{"role": "assistant", "content": revised_sent}]
                            )
                            result_str += revised_sent + " "
                        else:
                            chat_history.extend(
                                [{"role": "assistant", "content": init_new_sent}]
                            )
                            result_str += init_new_sent + " "
                        time_extrinsic = 0

                else:
                    if len(ex_list) > 0:
                        time_extrinsic += 1
                        ex_overall_list.extend(ex_list)                    

                        rewrited_sent = self.rewrite(ex_list, result_str, topic)
                        #rewrited_sent = self.delete(ex_list, result_str, topic)
                        rewrite_time_list[-1] += 1

                        in_list, ex_list = self.validate(rewrited_sent, result_str, topic=topic, self_query=self_query)
                        local_ex = 0
                        while (len(in_list) != 0 or len(ex_list) != 0) and local_ex < t_0:
                            local_ex += 1
                            ex_overall_list.extend(ex_list)             
                            #rewrited_sent = self.delete(ex_list, result_str, topic)
       
                            rewrited_sent = self.rewrite(ex_list, result_str, topic)
                            in_list, ex_list = self.validate(rewrited_sent, result_str, topic=topic, self_query=self_query)
                            rewrite_time_list[-1] += 1

                        if self.output_file:
                            with open(self.output_file, "a", buffering=1) as f:
                                print("rewrited:", rewrited_sent, file=f)
                                f.close()
                        chat_history.extend(
                                [{"role": "assistant", "content": rewrited_sent}]
                            )
                        result_str += rewrited_sent + " "
                    else:
                        if len(in_list) > 0:
                            revised_sent = self.revise(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                            if self.output_file:
                                with open(self.output_file, "a", buffering=1) as f:
                                    print("revised:", revised_sent, file=f)
                                    f.close()
                            revise_time_list[-1] += 1

                            chat_history.extend(
                                [{"role": "assistant", "content": revised_sent}]
                            )
                            result_str += revised_sent + " "
                        else:
                            chat_history.extend(
                                [{"role": "assistant", "content": init_new_sent}]
                            )
                            result_str += init_new_sent + " "

            else:
                if len(in_list) == 0:
                    chat_history.extend(
                        [{"role": "assistant", "content": init_new_sent}]
                    )
                    overall_extrinsic.extend(ex_list)
                    result_str += init_new_sent + " "

                else:
                    revised_sent = self.revise(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                    if self.output_file:
                        with open(self.output_file, "a", buffering=1) as f:

                            print("revised:", revised_sent, file=f)
                            f.close()
                    revise_time_list[-1] += 1

                    chat_history.extend(
                        [{"role": "assistant", "content": revised_sent}]
                    )

                    overall_extrinsic.extend(ex_list)
                    result_str += revised_sent + " "


        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"chat history: \n{chat_history}", file=f)
                f.close()
        
        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:

                print("done!!!!!", file=f)
                f.close()
        return result_str, self.openai_time, self.search_time, self.hf_time, list(self.links), ex_overall_list, list(self.evidence), revise_time_list, rewrite_time_list


    def generate(self, prompt, topic=None, temp=0.5, min_tokens=200, sent_max_tokens=128, ex_toler=0, discard=False, self_query=False):
        self.search_time = 0
        self.openai_time = 0
        self.hf_time = 0
        self.links = set()
        self.evidence = set()
        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"start=={topic}", file=f)
        result_str = ""
        template = ""
        #with open("demos/demo_bio_plm.txt", "r") as f:
            #template = f.read()
        template += prompt + "\nA: "
        chat_history = [     
            {"role": "system", "content": "You are a brilliant assistant to generate a human biography based on your knowledge. Please generate only one sentence at a time. Don't ask anything. "},
            {"role": "user", "content": template}
        ]
        overall_extrinsic = []
        time_extrinsic = 0
        ex_overall_list = []
        while (len(result_str.split(" ")) < min_tokens and time_extrinsic < 3):
            
            if not result_str:
                init_new_sent = self.sent_generate(chat_history, temp, sent_max_tokens, presence_penalty=0)
            else:
                init_new_sent = self.sent_generate(chat_history, temp, sent_max_tokens,presence_penalty=0.5)
            
            if "Tell me a bio of" in init_new_sent or len(init_new_sent) < 20:
                break
            if self.output_file:
                with open(self.output_file, "a", buffering=1) as f:
                    print("init:", init_new_sent, file=f)
                    f.close()
            in_list, ex_list = self.validate(init_new_sent, result_str, topic=topic, self_query=self_query)                

            if ex_toler == 0:
                if discard:
                    if len(ex_list) > 0:
                        time_extrinsic += 1
                        ex_overall_list.extend(ex_list)                    
                    else:
                        if len(in_list) > 0:
                            
                            revised_sent = self.revise_llama1(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)

                            #revised_sent = self.revise(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                            if self.output_file:
                                with open(self.output_file, "a", buffering=1) as f:
                                    print("revised:", revised_sent, file=f)
                                    f.close()

                            chat_history.extend(
                                [{"role": "assistant", "content": revised_sent}]
                            )
                            result_str += revised_sent + " "
                        else:
                            chat_history.extend(
                                [{"role": "assistant", "content": init_new_sent}]
                            )
                            result_str += init_new_sent + " "
                        time_extrinsic = 0

                else:
                    if len(ex_list) > 0:
                        time_extrinsic += 1
                        ex_overall_list.extend(ex_list)        

                        rewrited_sent = self.revise_llama1(init_new_sent, ex_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                        local_ex = 0
                        while (len(in_list) != 0 or len(ex_list) != 0) and local_ex < 0:
                            local_ex += 1
                            ex_overall_list.extend(ex_list)             
                            #rewrited_sent = self.delete(ex_list, result_str, topic)
       
                            rewrited_sent = self.rewrite(ex_list, result_str, topic)
                            in_list, ex_list = self.validate(rewrited_sent, result_str, topic=topic, self_query=self_query)

                        if self.output_file:
                            with open(self.output_file, "a", buffering=1) as f:
                                print("rewrited:", rewrited_sent, file=f)
                                f.close()
                        chat_history.extend(
                                [{"role": "assistant", "content": rewrited_sent}]
                            )
                        result_str += rewrited_sent + " "
                    else:
                        if len(in_list) > 0:
                            revised_sent = self.revise_llama1(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                            if self.output_file:
                                with open(self.output_file, "a", buffering=1) as f:
                                    print("revised:", revised_sent, file=f)
                                    f.close()

                            chat_history.extend(
                                [{"role": "assistant", "content": revised_sent}]
                            )
                            result_str += revised_sent + " "
                        else:
                            chat_history.extend(
                                [{"role": "assistant", "content": init_new_sent}]
                            )
                            result_str += init_new_sent + " "

            else:
                if len(in_list) == 0:
                    chat_history.extend(
                        [{"role": "assistant", "content": init_new_sent}]
                    )
                    overall_extrinsic.extend(ex_list)
                    result_str += init_new_sent + " "

                else:
                    revised_sent = self.revise_llama1(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)

                    #revised_sent = self.revise(init_new_sent, in_list, result_str, topic, max_tokens=sent_max_tokens, n=1)
                    if self.output_file:
                        with open(self.output_file, "a", buffering=1) as f:

                            print("revised:", revised_sent, file=f)
                            f.close()

                    chat_history.extend(
                        [{"role": "assistant", "content": revised_sent}]
                    )

                    overall_extrinsic.extend(ex_list)
                    result_str += revised_sent + " "


        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"chat history: \n{chat_history}", file=f)
                f.close()
        
        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:

                print("done!!!!!", file=f)
                f.close()
        return result_str, self.openai_time, self.search_time, self.hf_time, list(self.links), ex_overall_list, list(self.evidence)

    def user_warning(self, return_string, extrinsic_list):
        return_string += "\n\nWARNING: The following claims that have been generated might be not objectively correct:\n\n"
            
        for item in extrinsic_list:
            return_string += "Entity: {}\nClaim: {}\n\n".format(item["entity"], item["claim"])
        
        return_string += "Please note that as an AI model, I cannot guarantee the correctness of these facts. \nIt's essential not to rely solely on these without proper verification from trusted sources."
        
        return return_string

    def rewrite(self, extrinsic_list, context, topic, max_tokens=64, temp=0.1, n=1):
        #prompt = f"For the original text, which should be after the preceding context, please revise each wrong entity mentioned below, along with its associated wrong claim, based on the provided feedback.\n\nPreceding Context: {context}\n\nOriginal text: {sent}\n\n"
        if context:
            prompt = f"Given the evidence provided, craft a continuation sentence in response to the original user query 'Tell me a bio of {topic}'. Use the provided context solely for reference to maintain flow but AVOID REPETITION. Your response should only include the new sentence and exclude the preceding context.\n\nPreceding context: {context}\n\nEvidence:\n\n"

            #prompt = f"Please rewrite the original sentence about {topic} by deleting the unverifiable entities with the corresponding claims and reserving the verifiable entites with the corresponding claims. Please note that the preceding context is provided solely to ensure fluency. There's no need to change it. DO NOT INCLUDE ANY information from the preceding context in the revision, as redundancy is not desired. Provide only the rewrited version of the original text WITHOUT including the preceding context.\n\nPreceding context: {context}\n\nOriginal sentence: {sent}\n\n"
        else:
            #prompt = f"Please rewrite the original sentence about {topic} by deleting the unverifiable entities with the corresponding claims and reserving the verifiable entites with the corresponding claims. Provide only the rewrited version of the original sentence.\n\nOriginal sentence: {sent}\n\n"
            prompt = f"Given the evidence provided, craft a continuation sentence in response to the original user query 'Tell me a bio of {topic}'. Limit your response to a single sentence.\n\nEvidence:\n\n"


        #for ele in veri_list:
            #prompt += f"Verifiable entity: {ent}\nVerifiable claim: {claim}\n\n"

        for ele in extrinsic_list:
            #ent = ele["entity"]
            #claim = ele["claim"]
            evidence = ele["evidence"]
            #feedback = ele["feedback"]
            prompt += evidence + "\n"
            #prompt += f"Unverifiable entity: {ent}\nUnverifiable claim: {claim}\nRelated Facts: {evidence}\n\n"

        prompt += "\nBased on the evidence above, the continuation sentence should be:\n\n"

        message = [
                    {"role": "system", "content": "You are a briliant assistant. I want you to write a continuation sentence based on provided evidence and preceding contexts. PLEASE ONLY RETURN THE SENTENECE."},
                    {"role": "user", "content": prompt}
                ]
        
        if self.gen_model_name == "gpt-3.5-turbo" or self.gen_model_name == "gpt-4":
            revised_response = self.request_openai(message=message, n=n, temp=temp, model_name=self.gen_model_name, max_tokens=max_tokens)

            revised = revised_response.choices[0].message.content

        else:

            if not self.dola:
                revised_response = self.hf_gen(prompt, "gen",  n=n, temp=temp, max_new_tokens=100)
                revised = revised_response[0]["generated_text"][len(prompt):].strip()

            else:
                generate_kwargs = self.generate_kwargs
                generate_kwargs["temperature"] = temp
                generate_kwargs["max_new_tokens"] = 100
                revised = self.dola_gen(prompt, generate_kwargs)
            #if self.output_file:
                #print(revised, file=self.output_file)

            # Use nltk to split the output into sentences
            sentences = sent_tokenize(revised)
            
            if sentences:
                return sentences[0]
            else:
                return revised

        if "\n" in revised:
           revised = revised[:revised.index("\n")]
        return revised

    def delete(self, sent, extrinsic_list, context, topic, max_tokens=64, temp=0, n=1):
        #prompt = f"For the original text, which should be after the preceding context, please revise each wrong entity mentioned below, along with its associated wrong claim, based on the provided feedback.\n\nPreceding Context: {context}\n\nOriginal text: {sent}\n\n"
        if context:
            prompt = f"Referring to the related facts, please rewrite the original sentence about {topic} by deleting/correcting the unverifiable entities with the corresponding claims. Please note that the preceding context is provided solely to ensure fluency. There's no need to change it. DO NOT INCLUDE ANY information from the preceding context in the revision, as redundancy is not desired. Provide only the rewrited version of the original text WITHOUT including the preceding context.\n\nPreceding context: {context}\n\nOriginal sentence: {sent}\n\n"

            #prompt = f"Please rewrite the original sentence about {topic} by deleting the unverifiable entities with the corresponding claims and reserving the verifiable entites with the corresponding claims. Please note that the preceding context is provided solely to ensure fluency. There's no need to change it. DO NOT INCLUDE ANY information from the preceding context in the revision, as redundancy is not desired. Provide only the rewrited version of the original text WITHOUT including the preceding context.\n\nPreceding context: {context}\n\nOriginal sentence: {sent}\n\n"
        else:
            #prompt = f"Please rewrite the original sentence about {topic} by deleting the unverifiable entities with the corresponding claims and reserving the verifiable entites with the corresponding claims. Provide only the rewrited version of the original sentence.\n\nOriginal sentence: {sent}\n\n"
            prompt = f"Referring to the related facts, please rewrite the original sentence about {topic} by deleting the unverifiable entities with the corresponding claims. Provide only the rewrited version of the original sentence.\n\nOriginal sentence: {sent}\n\n"

        #for ele in veri_list:
            #prompt += f"Verifiable entity: {ent}\nVerifiable claim: {claim}\n\n"

        for ele in extrinsic_list:
            ent = ele["entity"]
            claim = ele["claim"]
            evidence = ele["evidence"]
            #feedback = ele["feedback"]
            prompt += f"Unverifiable entity: {ent}\nUnverifiable claim: {claim}\nRelated Facts: {evidence}\n\n"

        prompt += "Based on the related facts, the rewrited sentence would be:\n\n"

        message = [
                    {"role": "system", "content": "You are a briliant assistant. I want you to rewrite a sentencet to replace/correct the unverifiable factual entites based on related facts. PLEASE ONLY RETURN THE REWRITED SENTENECE."},
                    {"role": "user", "content": prompt}
                ]
        
        if self.gen_model_name == "gpt-3.5-turbo" or self.gen_model_name == "gpt-4" :
            revised_response = self.request_openai(message=message, n=n, temp=temp, model_name=self.gen_model_name, max_tokens=max_tokens)

            revised = revised_response.choices[0].message.content

        else:
            for dic in message:
                if dic["role"] == "system":
                    pt = PT(system_prompt=dic["content"])
                elif dic["role"] == "user":
                    pt.add_user_message(dic["content"])
                elif dic["role"] == "assistant":
                    pt.add_model_reply(dic["content"])
            prompt = pt.build_prompt()
            revised_response = self.hf_gen(prompt, "gen",  n=n, temp=temp, max_new_tokens=100)
            revised = revised_response[0]["generated_text"][len(prompt):].strip()

        if "\n" in revised:
           revised = revised[:revised.index("\n")]
        return revised

    def revise(self, sent, intrinsic_list, context, topic, max_tokens=64, temp=0.1, n=1):
        #prompt = f"For the original text, which should be after the preceding context, please revise each wrong entity mentioned below, along with its associated wrong claim, based on the provided feedback.\n\nPreceding Context: {context}\n\nOriginal text: {sent}\n\n"
        if context:
            prompt = f"Given the preceding context, please correct all wrong entities and their associated claims in the original text about {topic}, if you are confident the correct answer is in the evidence provided below. If not, you could also rewrite the sentence by deleting the wrong claim. Please note that the preceding context is provided solely to ensure fluency. There's no need to change it. DO NOT INCLUDE ANY information from the preceding context in the revision, as redundancy is not desired. Provide only the revised version of the original text WITHOUT including the preceding context.\n\nPreceding Context: {context}\n\nOriginal text: {sent}\n\n"
        else:
            prompt = f"Please correct all wrong entities and their associated claims in the original text about {topic}, if you are confident the correct answer is in the evidence provided below. If not, you could also rewrite the sentence by deleting the wrong claim. Provide only the revised version of the original text.\n\nOriginal text: {sent}\n\n"

        for ele in intrinsic_list:
            ent = ele["entity"]
            claim = ele["claim"]
            evidence = ele["evidence"]
            #feedback = ele["feedback"]
            prompt += f"Wrong entity: {ent}\nWrong claim: {claim}\nEvidence: {evidence}\n\n"

        prompt += "Revised original text: "


        message = [
            {"role": "system", "content": "You are a helpful assistant. Based on the provided evidence, you will rewrite the text to correct the wrong facts in the original text in ONE sentence."},
            {"role": "user", "content": prompt}
        ]
        if self.gen_model_name == "gpt-3.5-turbo":

            revised_response = self.request_openai(message=message, n=n, temp=temp, model_name=self.gen_model_name, max_tokens=max_tokens)

            revised = revised_response.choices[0].message.content
        
        else:
            if not self.dola:
                ret_list = self.hf_gen(prompt, "gen", temp=temp, max_new_tokens=max_tokens, n=n)
                revised = ret_list[0]['generated_text'][len(prompt):].strip()
            else:
                generate_kwargs = self.generate_kwargs
                generate_kwargs["temperature"] = 0.1
                generate_kwargs["max_new_tokens"] = max_tokens
                revised = self.dola_gen(prompt, generate_kwargs)
            # Use nltk to split the output into sentences
            sentences = sent_tokenize(revised)
            
            if sentences:
                return sentences[0]
            else:
                return revised
        if "\n" in revised:
           revised = revised[:revised.index("\n")]
        return revised

    def hf_gen(self, prompt, pipe, temp, max_new_tokens=64, n=1):
        eos_token_id = self.eval_tokenizer.encode("\n")[0]
        st = time.time()
        if pipe == "gen":
            if temp == 0:
                sequences = self.gen_pipeline(
                    prompt,
                    do_sample=False,
                    num_return_sequences=n,
                    eos_token_id=self.gen_tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.1
                )
            else:
                sequences = self.gen_pipeline(
                    prompt,
                    do_sample=True,
                    temperature=temp,
                    num_return_sequences=n,
                    eos_token_id=self.gen_tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.1

                )
        
        if pipe == "eval":
            if temp == 0:
                sequences = self.eval_pipeline(
                    prompt,
                    do_sample=False,
                    num_return_sequences=n,
                    eos_token_id=eos_token_id,
                    max_new_tokens=max_new_tokens,
                )
            else:
                sequences = self.eval_pipeline(
                    prompt,
                    do_sample=True,
                    temperature=temp,
                    num_return_sequences=n,
                    eos_token_id=eos_token_id,
                    max_new_tokens=max_new_tokens,
                )

        self.hf_time += time.time() - st

        return sequences

    def revise_llama1(self, sent, intrinsic_list, context, topic, max_tokens=64, temp=0, n=1):
        #prompt = f"For the original text, which should be after the preceding context, please revise each wrong entity mentioned below, along with its associated wrong claim, based on the provided feedback.\n\nPreceding Context: {context}\n\nOriginal text: {sent}\n\n"
        
        ent_str = ""
        evid_str = ""
        for ele in intrinsic_list:
            ent_str += ele["entity"] + "; "
            #claim = ele["claim"]
            evid_str += ele["evidence"] + " "
            #feedback = ele["feedback"]
            #prompt += f"Wrong entity: {ent}\nWrong claim: {claim}\nEvidence: {evidence}\n\n"
        #if context:
            #prompt = f"Given the preceding context, please correct all wrong entities and their associated claims in the original text about {topic}, if you are confident the correct answer is in the evidence provided below. If not, you could also rewrite the sentence by deleting the wrong claim. Please note that the preceding context is provided solely to ensure fluency. There's no need to change it. DO NOT INCLUDE ANY information from the preceding context in the revision, as redundancy is not desired. Provide only the revised version of the original text WITHOUT including the preceding context.\n\nPreceding Context: {context}\n\nOriginal text: {sent}\n\n"
        #else:
            #prompt = f"Please correct all wrong entities and their associated claims in the original text about {topic}, if you are confident the correct answer is in the evidence provided below. If not, you could also rewrite the sentence by deleting the wrong claim. Provide only the revised version of the original text.\n\nOriginal text: {sent}\n\n"
        with open("demos/demo_revise.txt", "r") as f:
            prompt = f.read()
        prompt += f"Original sentence: {sent}\nWrong entity: {ent_str}\nEvidence: {evid_str}\nInstruction: Based on the evidence above, correct the wrong entity in the original sentence about the bio of {topic}.\nRevised original sentence: "
    
        message = [
                {"role": "system", "content": "You are a helpful assistant. Based on the provided evidence, you will rewrite the text to correct the wrong facts in the original text in ONE sentence."},
                {"role": "user", "content": prompt}
        ]
        if self.gen_model_name == "gpt-3.5-turbo":

            revised_response = self.request_openai(message=message, n=n, temp=temp, model_name=self.gen_model_name, max_tokens=max_tokens)

            revised = revised_response.choices[0].message.content
        
        else:
            if not self.dola:
                ret_list = self.hf_gen(prompt, "gen", temp=temp, max_new_tokens=max_tokens, n=n)
                revised = ret_list[0]['generated_text'][len(prompt):].strip()
            else:
                generate_kwargs = self.generate_kwargs
                generate_kwargs["temperature"] = 0.1
                generate_kwargs["max_new_tokens"] = max_tokens
                revised = self.dola_gen(prompt, generate_kwargs)
            # Use nltk to split the output into sentences
            sentences = sent_tokenize(revised)
            
            if sentences:
                return sentences[0]
            else:
                return revised
        if "\n" in revised:
           revised = revised[:revised.index("\n")]
        return revised
    

    def sent_generate(self, history, temp=0, max_tokens=128, presence_penalty=0):
        """
        Just generate one sentence given the prompt.
        """
        if self.gen_model_name == "gpt-3.5-turbo":

            sent_gen = self.request_openai(message=history, n=1, temp=temp, max_tokens=max_tokens, model_name=self.gen_model_name, presence_penalty=presence_penalty)
            sent_gen_content = sent_gen.choices[0].message.content

        else:            
            prompt = ""
            for dic in history:
                if dic["role"] == "system":
                    #sent_pt = PT(system_prompt=dic["content"])
                    pass
                elif dic["role"] == "user":
                    prompt += dic['content']
                elif dic["role"] == "assistant":
                    if dic["content"].strip():
                        prompt += dic["content"].strip() + " "
                    #sent_pt.add_model_reply(dic["content"])
                    #sent_pt.add_user_message("Continue with one more sentence.")
            if not self.dola:
                ret_list = self.hf_gen(prompt, pipe="gen", temp=temp, max_new_tokens=max_tokens, n=1)
                sent_gen_content = ret_list[0]['generated_text'][len(prompt):].strip()
            else:
                generate_kwargs = self.generate_kwargs
                generate_kwargs["temperature"] = temp
                generate_kwargs["max_new_tokens"] = max_tokens
                sent_gen_content = self.dola_gen(prompt, generate_kwargs)
            if "\n" in sent_gen_content:
                sent_gen_content = sent_gen_content.split("\n")[0].strip()
                    
            #if self.output_file:
                #print(sent_gen_content, file=self.output_file)

            # Use nltk to split the output into sentences
        sentences = sent_tokenize(sent_gen_content)
        
        if sentences:
            return sentences[0]
        else:
            return sent_gen_content


        #return sent_gen_content

    def get_subject(self, doc):
        for token in doc:
            if ("subj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return str(doc[start:end])


    def gen_question(self, sent, topic, entity_set, temp=0, n=1):
        fdic = {
            "text": sent,
            "entities": []
        }
        with open("demos/demo_yesno.txt", "r", encoding="utf-8") as f:
            prompt = f.read()

        prompt += 'Preceding contxt: ""\nSentence: {}\nFor the above sentence about "{}", generate a yes/no question WITHOUT any pronouns about the entity of "{}". The question MUST contain the entity.\nQuestion: '

        def generate_question_for_entity(ent):
            if ent:
                filled_prompt = prompt.format(sent, topic, ent)
                history = [
                            {"role": "system", "content": "You are a question generation assistant given a sentence and a entity. Replace any pronouns with its original proper name."},
                            {"role": "user", "content": filled_prompt}
                        ]
                
                if self.val_model_name == "gpt-3.5-turbo" or self.val_model_name == "gpt-4":
                    gen_q_response = self.request_openai(message=history, n=n, temp=temp, max_tokens=100)
                
                    question = gen_q_response.choices[0].message.content.replace('"', "").replace("'", "")
                else:
                    if not self.dola:
                        gen_q_response = self.hf_gen(filled_prompt, "eval",  n=n, temp=temp, max_new_tokens=50)
                        question = gen_q_response[0]["generated_text"][len(filled_prompt):].strip().replace('"', "").replace("'", "")
                    else:
                        generate_kwargs = self.generate_kwargs
                        generate_kwargs["temperature"] = 0.01
                        generate_kwargs["max_new_tokens"] = 50
                        question = self.dola_gen(filled_prompt, generate_kwargs)    
                
                    if "\n" in question:
                        question = question[:question.index("\n")]

            return {
                    "entity": ent,
                    "question": question,
                    "atomic_fact": ""
                    }

        with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            results = executor.map(generate_question_for_entity, entity_set)

        for result in results:
            if result:
                fdic["entities"].append(result)

        return fdic

    def get_atomic_from_q(self, sent, qdic, topic, temp=0, n=1):
        with open("demos/demo_affirmtive.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
        prompt += "Question: {}\nFor the above question about '{}', rewrite it into an affirmative sentence.\nAnswer: "

        def process_entity(entity):
            if entity:
                q = entity["question"]
                filled_prompt = prompt.format(q, topic)
                history = [
                    {"role": "system", "content": "You are a helpful assistant to rewrite an interrogative sentence into an affirmative sentence."},
                    {"role": "user", "content": filled_prompt}
                ]

                if self.val_model_name == "gpt-3.5-turbo" or self.val_model_name == "gpt-4":
                    atomic_response = self.request_openai(message=history, n=n, temp=temp, max_tokens=64)
                    
                    atomic_fact = atomic_response.choices[0].message.content.replace('"', "").replace("'", "")
                
                else:

                    if not self.dola:
                        atomic_response = self.hf_gen(filled_prompt, "eval",  n=n, temp=temp, max_new_tokens=50)
                        atomic_fact = atomic_response[0]["generated_text"][len(filled_prompt):].strip().replace('"', "").replace("'", "")
                    else:
                        generate_kwargs = self.generate_kwargs
                        generate_kwargs["temperature"] = 0.5
                        generate_kwargs["max_new_tokens"] = 50
                        atomic_fact = self.dola_gen(filled_prompt, generate_kwargs)    
                    if "\n" in atomic_fact:
                        atomic_fact = atomic_fact[:atomic_fact.index("\n")]
                return atomic_fact
            return None

        with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            results = executor.map(process_entity, qdic["entities"])

        for i, atomic_fact in enumerate(results):
            if atomic_fact:
                qdic["entities"][i]["atomic_fact"] = atomic_fact

        return qdic

    def get_atomic_from_q22(self, sent, qdic, topic, temp=0, n=1):
        
        with open("demos/demo_affirmtive.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
        prompt += "Question: {}\n\For the above question about '{}', rewrite it into an affirmative sentence.\nAnswer: "

        for i in range(len(qdic["entities"])):
            if qdic["entities"][i]:
                q = qdic["entities"][i]["question"]
                filled_prompt = prompt.format(q, topic)
                history = [
                    {"role": "system", "content": "You are a helpful assistant to rewrite an interrogative sentence into an affirmative sentence."},
                    {"role": "user", "content": filled_prompt}
                ]

                if self.val_model_name == "gpt-3.5-turbo" or self.val_model_name == "gpt-4":
                    atomic_response = self.request_openai(message=history, n=n, temp=temp, max_tokens=64)
                    
                    atomic_fact = atomic_response.choices[0].message.content.replace('"', "").replace("'", "")
                
                else:

                    if not self.dola:
                        atomic_response = self.hf_gen(filled_prompt, "eval",  n=n, temp=temp, max_new_tokens=50)
                        atomic_fact = atomic_response[0]["generated_text"][len(filled_prompt):].strip().replace('"', "").replace("'", "")
                    else:
                        generate_kwargs = self.generate_kwargs
                        generate_kwargs["temperature"] = 0.5
                        generate_kwargs["max_new_tokens"] = 50
                        atomic_fact = self.dola_gen(filled_prompt, generate_kwargs)    
                    if "\n" in atomic_fact:
                        atomic_fact = atomic_fact[:atomic_fact.index("\n")]

                qdic["entities"][i]["atomic_fact"] = atomic_fact

        return qdic
            
    def process_element(self, ele):
        entity = ele["entity"]    
        claim = ele["claim"]         
        question = ele["question"]       
        evidence_list = ele["evidence"]  
        links_set = ele["links"]
        evidence = " ".join(evidence_list)
        res_fact = self.fact_check(question, evidence, temp=0.3, n=3)

        if res_fact == None:
            return ("extrinsic", {
                "entity": entity,
                "claim": claim,
                "question": question,
                "evidence": evidence
            })
        else:
            self.evidence.update(evidence_list)
            self.links.update(links_set)
            if not res_fact:
                return ("intrinsic", {
                    "entity": entity,
                    "claim": claim,
                    "question": question,
                    "evidence": evidence
                })
        return None

    def validate(self, sent, preceding="", topic=None, self_query=False):
        entity_set = self.extract_entity(sent, topic, temp=0, n=1)
        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"entity set:\n{entity_set}", file=f)
                f.close()
        
        question_dic = self.gen_question(sent, topic, entity_set, temp=0, n=1)

        fact_dic = self.get_atomic_from_q(sent, question_dic, topic, temp=0, n=1)
        """
        fact_dic = self.extract_facts(sent, preceding, entity_set, temp=0.5, n=2)
        """
        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"fact&question_dict:\n{fact_dic}", file=f)
                f.close()
       

        evidence_dict = self.group_search(fact_dic, topic, self_query)
        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print(f"evidence dic:\n{evidence_dict}", file=f)
                f.close()
        
        extrinsic_list = []
        intrinsic_list = []

        with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            results = executor.map(self.process_element, evidence_dict["atomic_fact"])

        for result in results:
            if result:
                if result[0] == "extrinsic":
                    extrinsic_list.append(result[1])
                elif result[0] == "intrinsic":
                    intrinsic_list.append(result[1])

        if self.output_file:
            with open(self.output_file, "a", buffering=1) as f:
                print("ex===", extrinsic_list, file=f)
                print("in===", intrinsic_list, file=f)

        return intrinsic_list, extrinsic_list

    def fact_check(self, claim, evidence, temp=1, n=5):
        #with open("fact_check_7b.txt", "a") as f:
            #print(claim, evidence, file=f)
        with open("demos/demo_fact-check_bio_short.txt", "r") as f:
            response_template = f.read()
            #response_template += "Evidence: {}\nClaim: {}\nAnswer: "
            response_template += "Evidence: {}\nQuestion: {}\nAnswer: "


        filled_prompt = response_template.format(evidence, claim)

        history = [
                    {"role": "system", "content": "You are a fact-checking assistant. Based on the provided evidence, you will check if the claim is true or false given the provided evidence."},

                    {"role": "user", "content": filled_prompt}
                ]
        
        if self.val_model_name == "gpt-3.5-turbo" or self.val_model_name == "gpt-4":

            factcheck_response = self.request_openai(message=history, n=n, temp=temp, max_tokens=100)
        
            factcheck_result_list = [factcheck_response.choices[i].message.content for i in range(n)]

        else:
            if not self.dola:
                factcheck_response = self.hf_gen(filled_prompt, "eval",  n=1, temp=1, max_new_tokens=256)
                factcheck_result_list = []
                for i in range(1):
                    res = factcheck_response[i]["generated_text"][len(filled_prompt):].strip()
                    if "\n" in res:
                        res = res.split("\n")[0].strip()
                    factcheck_result_list.append(res)
            else:
                generate_kwargs = self.generate_kwargs
                generate_kwargs["temperature"] = 0.01
                generate_kwargs["max_new_tokens"] = 256
                res = self.dola_gen(filled_prompt, generate_kwargs)   
                factcheck_result_list = []
                if "\n" in res:
                    res = res.split("\n")[0].strip()
                factcheck_result_list.append(res)  

        count_dic = {
            "true": 0,
            "false": 0,
            "not enough information": 0
        }
        fb_dic = {
            "true": [],
            "false": [],
            "not enough information": []
        }
        for res in factcheck_result_list:
            for key in count_dic:
                if key in res.lower():
                    count_dic[key] += 1
                    fb_dic[key].append(res)

        factcheck_result = max(count_dic, key=count_dic.get)
        #with open("fact_check_7b.txt", "a") as f:
            #print(claim, factcheck_result_list, count_dic, factcheck_result, file=f)

        if factcheck_result[-1] == ".":
            factcheck_result = factcheck_result[:-1]

        if "true" in factcheck_result:
            return True
        elif "false" in factcheck_result:
            return False
        elif "not enough information" in factcheck_result:
            return None
        return factcheck_result

    def single_atmoic_gpt3(self, context, entity, temp=1, n=3):
        with open("demos/demo_atomic_replace_single.txt", "r") as f:
            response_template = f.read()
            #response_template += "Evidence: {}\nClaim: {}\nBased on the evidence and claim above, the evidence is "
            response_template += "Context: {}\n\Standalone Fact: "

        filled_prompt = response_template.format(context, entity)

        message = [
                    {"role": "system", "content": "You are a assistant to help me complete a atomic fact for the factual key entity."},

                    {"role": "user", "content": filled_prompt}
                ]
        

        atomic_response = self.request_openai(message=message, n=n, temp=temp, max_tokens=128)
        
        atomic_list = list()
        for i in range(len(atomic_response.choices)):
            atomic_result = atomic_response.choices[i].message.content
            atomic_list.append(atomic_result)
        return atomic_list


    def extract_facts(self, sent, preceding, entity_set, temp=0, n=1):
        fdic = {
            "text": sent,
            "entities": []
        }
        for entity in entity_set:
            replaced_text = sent.replace(entity, "<entity>")
            res = self.single_atmoic_gpt3(preceding+" "+replaced_text, entity, temp=temp, n=n)

            filtered_set = set(res)

            #accepted_facts = process_facts(res, entity, subject)
            accepted_fact = max(filtered_set, key = res.count).replace("<entity>", entity)
            
            fdic["entities"].append({
                entity: accepted_fact
            })

        return fdic
    
    def single_search(self, query, subject):

        st = time.time()
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
        "q": query,
        "autocorrect": False,
        "hl": "en",
        "type": "search",
        })
        headers = {
        'X-API-KEY': self.__serper_key,
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        decoded = data.decode("utf-8")
        dic = json.loads(decoded)

        et = time.time()

        self.search_time = et-st
        knowledge_dict = {}
        evidence_list = []
        num_evidence = 2
        links = set()

        if 'answerBox' in dic.keys():
            if "answer" in dic["answerBox"]:
                if len(dic["answerBox"]["answer"].split()) > 5:
                    evidence_list.append(dic["answerBox"]["answer"])
                    if "link" in dic["answerBox"]:
                        links.add(dic["answerBox"]["link"])

            elif "snippet" in dic["answerBox"]:
                if len(dic["answerBox"]["snippet"].split()) > 5:
                    evidence_list.append(dic["answerBox"]["snippet"])
                    if "link" in dic["answerBox"]:
                        links.add(dic["answerBox"]["link"])

        if "knowledgeGraph" in dic.keys() and "description" in dic["knowledgeGraph"]:
            if len(dic["knowledgeGraph"]["description"].split()) > 5:
                evidence_list.append(dic["knowledgeGraph"]["description"])
                if "descriptionLink" in dic["knowledgeGraph"]:
                    links.add(dic["knowledgeGraph"]["descriptionLink"])

        if 'organic' in dic.keys():
            i = 0
            while len(dic["organic"]) > i and len(evidence_list) < num_evidence:
                evid = dic["organic"][i]["title"] + ": "+ dic["organic"][i]["snippet"]
                if any(word.lower() in evid.lower() for word in subject.split(" ")):
                    evidence_list.append(dic["organic"][i]["title"] + ": "+ dic["organic"][i]["snippet"])
                    links.add(dic["organic"][i]["link"])

                i += 1

        # could change the number of evidence used
        knowledge_dict = {}
        knowledge_dict["evidence"] = evidence_list
        knowledge_dict["links"] = links
        knowledge_dict["claim"] = query

        return knowledge_dict
    
    def is_relevant(self, query, evidence, entity, subject, threshold=0.2):
        """
        st = time.time()
        q_list = set(pos_tag_method(query.lower().replace(entity.lower(),"")))
        evid_list = set(pos_tag_method(evidence.lower().replace(entity.lower(),"")))
        ent_tokens = set(tokenize_process(entity))
        evid_tokens = set(tokenize_process(evidence))
        et = time.time()
        self.utils_time += et-st
        precision = len(q_list & evid_list) / len(q_list) if len(q_list) != 0 else 0
        #return True
        """
        ent_tokens = set(tokenize_process(entity))
        evid_tokens = set(tokenize_process(evidence))
        return subject.lower() in evidence.lower() and len(ent_tokens & evid_tokens) >= 1
    
    def group_search(self, fact_dic, subject, self_query=False):
        ret_dic = {
            "text": fact_dic["text"],
            "atomic_fact" : []
        }
        for i in range(len(fact_dic["entities"])):
            #entity, atomic_fact = list(fact_dic["entities"][i].items())[0]
            entity = fact_dic["entities"][i]["entity"]
            question = fact_dic["entities"][i]["question"]
            atomic_fact = fact_dic["entities"][i]["atomic_fact"]

            if not self_query:
                final_evidence = set()           
                links = set()
               #queries = self.search_query_suggestion(subject, fact_dic["text"], entity, n=1)
                #if self.output_file:
                    #with open(self.output_file, "a", buffering=1) as f:
                        #print("suggested_queries", file=f)
                        #print(queries, file=f)
                        #f.close()


                #for query in queries:

                    #retrived_dict = self.single_search(query, subject)
                    #final_evidence.update(retrived_dict["evidence"])
                    #links.update(retrived_dict["links"])

                # question search
                retrived_dict = self.single_search(question, subject)
                final_evidence.update(retrived_dict["evidence"])
                links.update(retrived_dict["links"])


                # fuzzy search:
                retrived_dict = self.single_search(atomic_fact, subject)
                final_evidence.update(retrived_dict["evidence"])
                links.update(retrived_dict["links"])


                #quote_subject:
                atomic_fact_q = atomic_fact.replace(subject, f'"{subject}"')
                retrived_dict = self.single_search(atomic_fact_q, subject)
                final_evidence.update(retrived_dict["evidence"])
                links.update(retrived_dict["links"])


                #keyword:
                st = time.time()
                keyword_q = tokenize_process(entity)
                et = time.time()
                self.utils_time += et-st

                keyword_q.append(subject)

                keyword_q_str = ''
                for word in keyword_q:
                    keyword_q_str += f'"{word}"; '
                
                keyword_q_str = keyword_q_str[:-2]

                #print(keyword_q_str)
                retrived_dict = self.single_search(keyword_q_str, subject)
                final_evidence.update(retrived_dict["evidence"])
                links.update(retrived_dict["links"])


            else:
                with open("demos/demo_self_query_evidence.txt", "r") as f:
                    response_template = f.read()
                    response_template += "Question: {}\nAnswer: According to Wikipedia, "
                filled_prompt = response_template.format(question)

                history = [
                            {"role": "user", "content": filled_prompt}
                        ]
                
                if self.val_model_name == "gpt-3.5-turbo" or self.val_model_name == "gpt-4":

                    factcheck_response = self.request_openai(message=history, n=1, temp=0, max_tokens=200)
                
                    final_evidence = set([factcheck_response.choices[i].message.content for i in range(1)])

                else:
                    for dic in history:
                        if dic["role"] == "user":
                            prompt = dic["content"]
                    if not self.dola:
                        factcheck_response = self.hf_gen(prompt, "eval",  n=1, temp=0, max_new_tokens=200)
                        final_evidence = factcheck_response[0]["generated_text"][len(prompt):]
                    else:
                        generate_kwargs = self.generate_kwargs
                        generate_kwargs["temperature"] = 0.01
                        generate_kwargs["max_new_tokens"] = 256
                        final_evidence = self.dola_gen(filled_prompt, generate_kwargs)   
            
                    if "\n" in final_evidence:
                        final_evidence = set([final_evidence.split("\n")[0].strip()])
                    else:
                        final_evidence = set([final_evidence.strip()])

                links = set()

            ret_dic["atomic_fact"].append({
                "entity": entity,
                "question": question,
                "claim": atomic_fact,
                "evidence": final_evidence,
                "links": links
            })


        return ret_dic

    def search_query_suggestion(self, subject, entity, context, temp=0.5, n=1):
        prompt = f'For the entity "{entity}" in context "{context}" related to "{subject}", generate a short Google search query to verify its accuracy.\n\nQuery: '
        
        message = [
                    {"role": "system", "content": "You are a assistant to help me come up with a effective search query. Return the query only."},

                    {"role": "user", "content": prompt}
                ]

        if self.val_model_name == "gpt-3.5-turbo" or self.val_model_name == "gpt-4":
            query_response = self.request_openai(message=message, n=n, temp=temp, max_tokens=64)
            ret = [query_response.choices[i].message.content for i in range(n)]

        else:
            for dic in message:
                if dic["role"] == "user":
                    prompt = dic["content"]
            #prompt = pt.build_prompt()
            query_response = self.hf_gen(prompt, "eval",  n=n, temp=temp, max_new_tokens=10)
            ret = [query_response[i]["generated_text"][len(prompt):].strip() for i in range(n)]

        return ret

    def extract_entity_zero(self, sent, topic, temp=0.5, n=3):

        message = [
                    {"role": "system", "content": "You will be provided with a block of text, and your task is to extract a semicolon-separated list of keywords from it. "},

                    {"role": "user", "content": sent}
                ]
        extract_set = set()

        extract_response = self.request_openai(message=message, max_tokens=128, n=n, temp=temp)
        for i in range(len(extract_response.choices)):
            extract_result = extract_response.choices[i].message.content
            extract_set.update([ent.strip() for ent in extract_result.split(";") if topic.lower() not in ent])

        return extract_set
        

    def extract_entity(self, sent, topic, temp=0, n=1):
        with open("demos/demo_extract_bio_short.txt", "r", encoding="utf-8") as f:
            response_template = f.read()
            response_template += "Sentence: {}\nAnswer: "
        #response_template = "Sentence: {}\n\n"
        #response_template += "Identify all the important keyphrases from the above sentence and return a semicolon separated list."
        filled_prompt = response_template.format(sent)
        message = [
                    {"role": "system", "content": "You are a assistant to help me extract all important factual key phrases from a provided sentence."},

                    {"role": "user", "content": filled_prompt}
                ]
        
        if self.val_model_name == "gpt-3.5-turbo" or self.val_model_name == "gpt-4":
            extract_response = self.request_openai(message=message, max_tokens=128, n=n, temp=temp)
            extract_set = set()
            for i in range(len(extract_response.choices)):
                extract_result = extract_response.choices[i].message.content
                extract_set.update(extract_result.split(";"))
            extract_set = [ent.strip() for ent in extract_set if not sent[:len(ent)] == ent and (ent in sent and topic.lower() not in ent.lower())]
        
            # 1. Remove entities that are supstrings of another entity
            extract_set = sorted(extract_set, key=len)  # Sort by length
            to_remove = set()

            for i, ent in enumerate(extract_set):
                for j in range(i+1, len(extract_set)):
                    if ent in extract_set[j]:
                        to_remove.add(extract_set[j])

            extract_set = [ent for ent in extract_set if ent not in to_remove]

            # 2. only one entity remains for multiple entities with the same stemmed strings
            seen = set()
            filtered_union = []

            for ent in extract_set:
                st = time.time()
                stemmed_ent = stem(ent)
                et = time.time()
                self.utils_time += et-st
                if stemmed_ent not in seen and ent:
                    seen.add(stemmed_ent)
                    filtered_union.append(ent)
            extract_set = filtered_union

        else:
            for dic in message:
                if dic["role"] == "user":
                    prompt = dic["content"]
            if not self.dola:
                extract_response = self.hf_gen(prompt, "eval",  n=n, temp=temp, max_new_tokens=30)
                extract_set = set()
                for i in range(len(extract_response)):
                    extract_result = extract_response[i]["generated_text"][len(prompt):].strip()
                    if self.output_file:
                        with open(self.output_file, "a", buffering=1) as f:
                            print(f"extract output:", extract_result, file=f)
                    if "\n" in extract_result :
                        line = extract_result.split("\n")[0]
                        if ";" in line:
                            extract_result = line
                            extract_set.update(extract_result.split(";"))
                    else:
                        extract_set.update(extract_result.split(";"))
                    
            else:
                generate_kwargs = self.generate_kwargs
                generate_kwargs["temperature"] = 0.01
                generate_kwargs["max_new_tokens"] = 30
                extract_response = self.dola_gen(filled_prompt, generate_kwargs)   
                extract_response = [extract_response]   
                extract_set = set()
                for i in range(len(extract_response)):
                    extract_result = extract_response[i].strip()
                    if self.output_file:
                        with open(self.output_file, "a", buffering=1) as f:
                            print(f"extract output:", extract_result, file=f)
                    if "\n" in extract_result :
                        line = extract_result.split("\n")[0]
                        if ";" in line:
                            extract_result = line
                            extract_set.update(extract_result.split(";"))
                    #if self.output_file:
                        #with open(self.output_file, "a", buffering=1) as f:
                            #print(extract_result, file=f)
                            #f.close()
                    else:
                        extract_set.update(extract_result.split(";"))
            extract_set = [ent.strip() for ent in extract_set if topic.lower() not in ent.lower() and ent]

        return extract_set
