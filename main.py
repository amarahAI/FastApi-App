import uvicorn
from fastapi import FastAPI, Response, status, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import sys
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import (
	ChatPromptTemplate,
	HumanMessagePromptTemplate,
	MessagesPlaceholder,
)
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

from langchain.llms import WatsonxLLM


app = FastAPI(openapi_version="3.0.3",)
app.openapi_version = "3.0.3"

@app.get("/")
def watson_ce_root():
	return Response(content='Welcome to my project', status_code=status.HTTP_200_OK)

def preProcessLlmOutput(raw_string):
	cleanedResponse = raw_string.replace("Human","")
	cleanedResponse = cleanedResponse.replace("Chatbot","")
	cleanedResponse = cleanedResponse.replace("AI:","")
	cleanedResponse = cleanedResponse.replace("\n","")
	cleanedResponse = cleanedResponse.strip()
	return cleanedResponse


load_dotenv()
api_key_env = os.getenv("WATSONX_APIKEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
wx_project_id = os.getenv("WX_PROJECT_ID", None)


######### MODEL ############
print ('\n')
print ("Watson X credentials \n",api_key_env)
print (ibm_cloud_url)
print (wx_project_id)

parameters = {
	GenParams.DECODING_METHOD: "greedy",
	GenParams.MAX_NEW_TOKENS: 480,
	GenParams.MIN_NEW_TOKENS: 1,
	GenParams.TEMPERATURE: 0,
	GenParams.STOP_SEQUENCES: ['\n\n']
}

watsonx_llm = WatsonxLLM(
	model_id="ibm/granite-13b-chat-v2",
	url=ibm_cloud_url,
	project_id=wx_project_id,
	params=parameters,
)

watson_answer_prompt_start = """\nAgent: Okay, I am awaiting your instructions
\n\n
User:Watson, here are your instructions:
1. You are provided several documents.
2. You should generate response only using the information available in the documents.
3. If you can't find an answer, say \"I don't know\".
4. Do not use any other knowledge.
5. Summarise the response in precise manner and also don’t ask the questions .
6. Close the conversation with no further questions like system generated User questions.
7. While answering consider Dialysis as Hospitalization treatment and answer accordingly by giving explanation from document.

"""
watson_answer_prompt_end = """
Use the above context to improve the accuracy of the results.


\n\n
Agent:I am ready to answer your questions from the document. I will not repeat
answers I have given.
\n\n
User:{human_input}.

\n\n
Agent:"""


######## PROMPT ############
prompt = ChatPromptTemplate.from_messages(
[
	SystemMessage(
		# content="You are a chatbot having a conversation with a human."
		content=watson_answer_prompt_start
	),  # The persistent system prompt
	MessagesPlaceholder(
		variable_name="chat_history"
	),  # Where the memory will be stored.
	HumanMessagePromptTemplate.from_template(
		watson_answer_prompt_end
	),  # Where the human input will injected
]
)

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2, return_messages=True)


##### CHAIN #########
chat_llm_chain = LLMChain(
	llm=watsonx_llm,
	prompt=prompt,
	verbose=True,
	memory=memory,
)



class ResponseCustom(BaseModel):
	rag_response: str
	dummy_one: str
	dummy_two : str
	
class Query(BaseModel):
	query: str
@app.post("/get_model_response", response_model=ResponseCustom)
async def get_model_response(query: Query):
# def get_model_response_etxt(query):
	user_query = str(query).split('=')[1]
	print ("\nUser query:", user_query)

	##### CONTEXT ID ########
	similarity_status = True

	
	if(similarity_status):
		#Retain memory
		pass
	else:
		#Reset Memory
		print("Memory reset!")
		memory.clear()
		
	
	##### INVOKING ##########
	
	#Retreiver call
	# try:
	# 	response_wd = wdObj.wd_query_collection(str(user_query))
	# except:
	# 	print ("Discovery query exception")
	# 	sys.exit(0)

	context_text = ""
	# if(response_wd.status_code == 200):
	# 	resp_result = response_wd.get_result()
	# 	for item in resp_result["results"]:
	# 		context_text += str(item["document_passages"][0]["passage_text"])
	# 	print ("\n Context:", str(context_text))
	# else:
	# 	print ("WD Exception")

	#Context update
	memory.save_context({"input": "Context"}, {"output": str(context_text)})#.replace('[', '').replace(']', ''))})
	discovery_input = context_text
	inputs = {"human_input": user_query}

	##LLM call
	retval = chat_llm_chain.invoke(inputs)
	bot_response = preProcessLlmOutput(retval["text"])
	print ("\tHUMAN:", retval["human_input"])
	print ("\tBOT:", bot_response)
	
	return {"rag_response":str(bot_response), "dummy_one":"dummy_one", "dummy_two":"dummy_two"}


# Get the PORT from environment
# port = os.getenv('PORT', '8002')
server_port = 8000
if __name__ == "__main__":
	uvicorn.run( app, port=server_port)
	
