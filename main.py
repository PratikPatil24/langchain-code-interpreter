from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate


load_dotenv()


def main():
    print("Start")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    # base_prompt = hub.pull("langchain-ai/react-agent-template")
    # prompt = base_prompt.partial(instructions=instructions)
    base_prompt_template = """
    {instructions}
    
    TOOLS:
    ------
    
    You have access to the following tools:
    
    {tools}
    
    To use a tool, please use the following format:
    
    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```
    
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
    
    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```
    
    Begin!
    
    New input: {input}
    {agent_scratchpad}"""
    prompt = PromptTemplate(template=base_prompt_template).partial(
        instructions=instructions
    )
    print(prompt)

    tools = [PythonREPLTool()]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent = create_react_agent(prompt=prompt, llm=llm, tools=tools)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    res = agent_executor.invoke(
        {
            "input": """generate and save in current working directory 15 QRcodes
                                that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
        }
    )
    print(res)


if __name__ == "__main__":
    main()
