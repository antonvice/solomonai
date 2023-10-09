import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import ChatOpenAI
from agunt import *
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    # Create the Agree button action
    actions = [
        cl.Action(name="Agree", value="agree", description="Click to agree")
    ]
    
    # Send a message with the Agree button
    await cl.Message("Do you agree to the terms and conditions?", actions=actions).send()
    

@cl.action_callback("Agree")
async def start():
    verbose = True
    llm = ChatOpenAI(temperature=0)
    # Agent characteristics - can be modified
    config = dict(
        legal_person_name="Solomon",
        legal_person_role="Legal Representative",
        company_name="Solomon AI",
        company_business="Solomon AI is a premium legal services company that provides customers with the most detailed and easy way to form a positional paper",
        company_values="Our mission at Solomon AIn is to help people achieve hassle-free proposal paper creation",
        conversation_purpose="create a proposal paper for the customer",
        conversation_history=[],
        conversation_type="chat",
        conversation_stage=conversation_stages.get(
            "1",
            "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional.",
        ),
        use_tools=True,
        product_catalog="user_data.txt",
    )
    legal_agent = LegalGPT.from_llm(llm, verbose=False, **config)
    legal_agent.seed_agent()
    cl.user_session.set("llm_chain", legal_agent)

@cl.on_message
async def main(message):
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    print("Message", message,"CHAIN",llm_chain)
    llm_chain.human_step(message)
    llm_chain.determine_conversation_stage()
    res = llm_chain.step()
    
    await cl.Message(content=res).send()