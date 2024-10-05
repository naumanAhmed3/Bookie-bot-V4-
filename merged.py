from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import (AgentExecutor , create_react_agent , create_structured_chat_agent)
import requests
from langchain import hub
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage
import json
from datetime import datetime
from cmd_version import staticTool
import streamlit as st 


model = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key="AIzaSyDvbke4TODM1nOMbkZAXXhOVGQeECSsATU")
memory = ConversationBufferMemory(memory_key="chat_history" ,return_messages=True) #Maintains the conversational context


@tool
def today_date():
    """
    Use this tool to know what date it is today.
    
    """
    
    today = datetime.today()
    # Format the date as a string (optional)
    today_str = today.strftime('%Y-%m-%d')  # Example: '2024-09-27'
    return today_str


@tool
def get_auth_token():
    """

    Use this tool to get the authentication token or when you get Error: 401 {"message":"Unauthenticated."}
    No arguments or action-inputs needed to use this.
    
    
    """
    # URL of the API endpoint
    url = 'https://bookmesky.com/partner/api/auth/token'
    
    # Headers to send with the request
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Hardcoded data for the POST request
    data = {
        'username': 'bookme-sky',
        'password': 'omi@work321'
    }
    
    # Make the POST request
    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful
    if response.status_code == 201:
        # Return the JSON response if the request was successful
        print(response.json())
        return response.json()
    else:
        # Print an error message if the request failed
        print(f"Error: {response.status_code}")
        memory.add_message(AIMessage(content=response.text))
        print(response.text)
        return None
    

@tool
def get_airlines(auth_token , depart_from_city , arrival_at_city , TravelClass):
    """
   
   Use this tool to get all the airlines that are offering the user's demanded flight.
   Use IATA codes for cities.

    Example:
    "action": "get_airlines",
    "action_input": 
{ "auth_token": "19799616|4Kd2rHNJZDW6g8JBiN5N4qTXM7uzdJVlNuaZOmkp05e302c8",
    "depart_from_city": "LHE",
    "arrival_at_city": "KHI",
    "TravelClass": "economy"}

    """
    # URL of the API endpoint
    url = 'https://bookmesky.com/air/api/content-providers'
    
    # Headers for the request
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {auth_token}'  # Bearer token for Authorization
    }
    
    # Data to be sent in the POST request
    data = {
        "Locations": [
            {"IATA": depart_from_city, "Type": "airport"},
            {"IATA": arrival_at_city, "Type": "airport"}
        ],
        "TravelClass": TravelClass
    }
    
    # Make the POST request
    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        memory.chat_memory.add_message(AIMessage(content=response.text))
        return response.json()  # Return the JSON response
    
    else:
        # Print an error message and return None if the request failed
        print(f"Error: {response.status_code}")
        print(response.text)
        return None



@tool
def search_flights(auth_token, depart_from_city, arrival_at_city, ContentProvider, TravelClass,
                  depart_date, arrive_date, adult_count, child_count, infant_count , trip_type=None):
    """
    
    Use this Tool to search for flight details.
    TravelClass parameter that is passed to API must be all lowercase e.g. "economy" not "Economy".
    Action Inputs are auth_token, depart_from_city, arrival_at_city, ContenProvider , TravelClass,
    depart_date, arrive_date, adult_count, child_count, infant_count. 
    Use these paramters with exactly same name.
    When user mentions "airline" they mean ContenProvider.
    Only one value can be passed in ContentProvider parameter
    When telling dates to user , tell them in an easy and readable format.
    Nothing must be pased to "ContentProvider" parameter other than airlines returned by "get_airlines" tool.
    Use TravelClass as "economy" if not mentioned by user , but inform them that the flight you searched for in economy class.
    Inform the user what response was returned for each API call and with what parameters.
    

    Search for "one_way" trip_type if this information is not provided in User's Quer and inform the user that you are searching for one way flights.
    In case of "one_way" trip_type set arrive_date parameter to "null".
    If the user mentions "return" trip_type then them arrive_date as well.
    Don't ask the user to provide date in a specific format , instead interpret the dates for parameters yourself from user's query in fromat YYYY-MM-DD. 
    

    """
    if trip_type:
    
        if trip_type == 'return' :
            dates= [depart_date , arrive_date]
        else:
            dates=[depart_date]
        url = 'https://bookmesky.com/air/api/search'
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {auth_token}'  }
        
        data = {
            "Locations": [
                {"IATA": depart_from_city, "Type": "airport"},
                {"IATA": arrival_at_city, "Type": "airport"}
            ],
            "ContentProvider": ContentProvider,
            "Currency": "PKR",
            "TravelClass":TravelClass,
            "TripType": trip_type,
            "TravelingDates": [depart_date],
            "Travelers": [
                {"Type": "adult", "Count": adult_count},
                {"Type": "child", "Count": child_count},
                {"Type": "infant", "Count": infant_count}
            ]
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        memory.chat_memory.add_message(AIMessage(content=response.text))
        
        return response.json()




tools =[
    get_auth_token, 
    get_airlines , 
    search_flights,
    today_date,
    staticTool
]


#Setting up buffer for LLM to have a look at.
role =""" 1)You are BookieBee, an agent-based chatbot for Bookme.pk, designed to assist users with a variety of queries related to services offered by the company. Bookme.pk provides services including bus ticket booking, cinema ticketing, airline ticket reservations, and event ticket sales. Use the available tools to provide accurate, up-to-date, and helpful 
            responses regarding these services, ensuring a seamless customer experience.
            Even if the query isn't regrading the services , try your best to answer it using "StaticTool".
          2)Try to figure out inputs/arguments for the tools you need to use from the user's given query and if they are not found in the query you must ask them from user.
          3)Use current date and time as a reference to identify dates/times from user's queries. To get sure about current date/time use the provided tool "today_date".
          4)Always introduce yourself as BookieBee, making the conversation engaging and friendly.
          5)Use decent and professional emojis where appropriate to make interactions more appealing and light-hearted.
          6)Format the your responses to make them easily readable and understandable.
                    
                   
      """
memory.chat_memory.add_message(SystemMessage(content=role))

agent = create_structured_chat_agent(
    llm=model , 
    tools=tools,
    prompt=hub.pull('hwchase17/structured-chat-agent'),
    stop_sequence=True 
    )

executor = AgentExecutor.from_agent_and_tools(
    agent = agent ,
    tools =tools ,
    verbose = True ,
    memory=memory , 
    max_execution_time=30000 , 
    handle_parsing_errors=True
)









