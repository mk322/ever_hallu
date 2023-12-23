### Ever: Mitigating Hallucination in Large Language Models through Real-Time Verification and Rectification

### How to run EVER?
First, you may need to have a Serper API key. Once you have the API key, pass it into the following argument to initialize a EVER instance. Also, don't forget to store your OpenAI API Key as your environment variable. 
```Python
prompt = "Tell me a bio of Jorge Enríquez."

pipeline = fact_gen(serper_api, gen_model="gpt-3.5-turbo", eval_model="gpt-3.5-turbo", workers_num=16)
```
Run the non-retrieval + evidence retrieval version:
```Python
res_str, openai_time, search_time, hf_time, source_list, extrinsic_dic, evidence_list, revise_list, rewrite_list = pipeline.generate(prompt, name, min_tokens=100, sent_max_tokens=128)
print(res_str)
```

Run the retrieval-augmented generation + evidence retrieval version:
```Python
res_str, openai_time, search_time, hf_time, source_list, extrinsic_dic, evidence_list, revise_list, rewrite_list = pipeline.init_generate(prompt, name, min_tokens=100, sent_max_tokens=128)
print(res_str)
```
