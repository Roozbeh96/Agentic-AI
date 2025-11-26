from langchain_ollama import ChatOllama


def test_func():

    llm = ChatOllama(model = "llama3.1",
                     temperature = 0.0,
                     base_url="http://127.0.0.1:11434")
    
    prompt = "You are a helpful assistant. Answer briefly: what is 2 times 3?"

    response = llm.invoke(prompt)
    print("Model reply:")
    print(response.content)

    

sample_ = test_func()