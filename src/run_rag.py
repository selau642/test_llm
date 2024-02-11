from src.llm_model import load_4bit_mistral_7b_instruct_v01, setup_hf_pipeline
from src.load_data import load_from_url, load_faiss_retriever
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

if __name__ == "__main__":
    url_list = [
        'https://cnaluxury.channelnewsasia.com/experiences/pizza-pop-singapore-diego-vitagliano-la-bottega-enoteca-antonio-miscellaneo-243026'
    ]
    """
    Heads up, pizza fans. 
    The ultimate in Italian pizzas will be in town later this month. 
    The world’s top pizzaiolo Diego Vitagliano will be serving 
    his famed pies at Italian restaurant La Bottega Enoteca 
    from Feb 20 (Tue) till Feb 23 (Fri).

    La Bottega Enoteca’s owner, Antonio Miscellaneo, 
    is no slouch in the pizza rankings either. He sits at 
    57th in the list of World’s Best Pizzas and his pies are 
    ranked 19th in the Asia-Pacific. But at this dinner, he will cede 
    the pizza-making to Vitagliano and instead contribute an appetiser, 
    main courses and dessert. “I wouldn’t want to make pizza 
    while Diego is here,” he said, laughing. “I’m only 
    integrating into Diego’s menu so that we are not 
    (just serving) pizza. I would love to serve eight courses of pizza, 
    but I think that would be too much for most people.”
    Vitagliano, whose pizzas are ranked number one in 
    the Italy-based Top Pizza World 2023, met Miscellaneo through 
    the event, which he describes as “not only a ranking, 
    but a community through which pizza makers of the 
    world meet and share, compare our jobs, and have fun together”.
    Diego Vitagliano. (Photo: Vitagliano)

    Vitagliano is often credited as one of the innovators 
    of modern Neapolitan pizza, having spent the 
    last decade perfecting his crusts with a focus on 
    lightness and digestibility. “Pizza is democratic,” 
    he declared, an ethos embodied in the options available 
    at his pizzerias in Naples. “At our pizzerias, everyone 
    can ask for exactly the pizza you want — thin crust, 
    crunchy, double-cooked. Pizza is the most democratic food 
    in the world,” he said through his interpreter during 
    a video interview with CNALuxury.

    For Miscellaneo, the event is a priceless learning 
    experience. “I invited Diego to Singapore because I’m 
    interested in learning from the best,” he said. “I feel 
    like there are very few of us (Italian pizzaiolos) 
    in Singapore and not enough of a community 
    to learn from one another.
    Pizza Cassula Fire. (Photo: Vitagliano)

    “Also, I wanted to bring the best of Italy to Singapore. 
    I think that Singapore is now at a point where people 
    can appreciate what Diego does. Five years ago, maybe not. 
    That’s why I’m happy that he’s coming.”

    To be sure, just five years ago, there was barely a 
    handful of Neapolitan pizza purveyors in the city. 
    It wasn’t until 2019, when Miscellaneo established his 
    wildly popular Casa Nostra private dining experience 
    serving airy Neapolitan-style pizzas, that the dining 
    public began paying attention to the quotidian Italian pie.

    “(Over) the last five years, there’s been a 
    blooming of pizzerias doing more Italian style of pizza, 
    whether Neapolitan or more contemporary Neapolitan. 
    More people have been exposed to it, to this type of crust… 
    maybe not to the flavours that Diego is going to do, 
    but that’s why it’s going to be good,” he added.
    Antonio Miscellaneo. (Photo: La Bottega)

    Among the ingredients that Vitagliano will 
    bring with him are Piennolo tomatoes 
    (an ancient variety grown on the slopes of Mount Vesuvius), 
    orsino garlic (wild Italian garlic) and peperone crusco 
    (sweet and crunchy dried peppers cultivated in Basilicata) 
    for his pizzas.

    “Certain things (like Italian vegetables) are difficult 
    to get in Singapore, so we had to find recipes 
    that can be reproduced here without compromise. 
    I don’t like to do something that’s not Italian, 
    so what we are doing is selecting Italian recipes 
    that are bolder in flavour (to suit the 
    Singaporean palate),” said Miscellaneo.

    At the time of writing, only a few seats 
    are still available for Feb 20. To sign up, go to La Bottega's website.
    """
    tokenizer, model = load_4bit_mistral_7b_instruct_v01()
    llm = setup_hf_pipeline(tokenizer, model)
    chunk_text_doc_list = load_from_url(url_list)
    faiss_retriever = load_faiss_retriever(chunk_text_doc_list)

    question = "What is so interesting about Diego"
    prompt = PromptTemplate(
        input_variables=['context', 'question'],    
        template="""
        ### [INST] Read the following passage and answer the question:
        Context: {context}
        Question: {question}
        [/INST]
        ###
        """
    )
    
    rag_chain = {
        "context": faiss_retriever,
        "question": RunnablePassthrough()
    } | prompt | llm     

    result = rag_chain.invoke("What is so interesting about Diego?")


    print(result)
    """
    Diego Vitagliano is a renowned Italian pizzaiolo 
    who has been recognized as the number one pizzaiolo 
    in the world. He is known for his innovative 
    approach to Neapolitan pizza, focusing on lightness 
    and digestibility. He has spent the past decade perfecting 
    his crusts and has established several pizzerias in Naples.
        
    For Antonio Miscellaneo, Diego's visit to Singapore is a 
    valuable learning experience. He feels that there are 
    not enough Italian pizzaiolos in Singapore and wants to 
    learn from the best. He also wants to bring the 
    best of Italy to Singapore and 
    showcase the diversity of Italian cuisine.
    """