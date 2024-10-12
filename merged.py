from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import (AgentExecutor , create_react_agent , create_structured_chat_agent)
import requests
from langchain import hub
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage
import json
from datetime import datetime , date
from cmd_version import staticTool
import streamlit as st 


model = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key="AIzaSyDvbke4TODM1nOMbkZAXXhOVGQeECSsATU")
import os

LLM_api = os.getenv('GOOGLE_API_KEY')
model = GoogleGenerativeAI(model='gemini-1.5-pro', google_api_key=LLM_api)
memory = ConversationBufferMemory(memory_key="chat_history" ,return_messages=True) #Maintains the conversational context


@tool
def today_date():
    """
    Use this tool to know what date it is today.
    
    """
    
    today = date.today()
    day_name = today.strftime("%A")

    # Format the date as a string (optional)
    return f"Today's date is {today} and day is {day_name}."


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


import requests
import json
from langchain_core.messages import AIMessage
from datetime import datetime

import requests
import json
from langchain_core.messages import AIMessage
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

@tool
def search_flights(auth_token, depart_from_city, arrival_at_city, ContentProvider=None, TravelClass='economy',
                   depart_date=None, arrive_date=None, adult_count=1, child_count=0, infant_count=0, trip_type='one_way'):
    """
    Use this Tool to search for flight details. The TravelClass parameter must be all lowercase (e.g., "economy").
    Use parameters: auth_token, depart_from_city, arrival_at_city, ContentProvider, TravelClass, 
    depart_date, arrive_date, adult_count, child_count, infant_count, trip_type.

    - All parameters must be passed while calling the API.

    FOR EXAMPLE : 
    {
    "action": "search_flights",
    "action_input": {
    "auth_token": "27086327|hPIW9kGD2KINLifBoMszkL62DcVUqNYmiW1Iu33Y737c129e",
    "depart_from_city": "LHE",
    "arrival_at_city": "KHI",
    "ContentProvider": "airblue",
    "depart_date": "2024-10-12",
    "TravelClass": "economy",
    "adult_count": 1,
    "child_count": 0,
    "infant_count": 0
                                }
                                            }
    
    - Must get the 'auth_token' parameter value using 'get_auth_token' tool before using this tool.
    - Don't introduce yourself again and again.
    - If 'trip_type' is not provided, default to 'one_way' and set 'arrive_date' to None.
    - If 'ContentProvider'/airline  is not specified in user's query ,must not ask it from user but call the API for all of the following airlines as value for 'ContentProvider' argument : airblue , airsial , sereneair , jazeera , flydubai , salamair , bookme-legacy , oneapi.​ 
    - If 'TravelClass' is not provided, default to 'economy'
    - The TravelClass parameter must be all lowercase (e.g., "economy" even if given as "Economy" by user ) .
    - For 'return' trips, set both depart_date and arrive_date.
    - For date refernces must use the today_date tool.
    - Automatically interpret dates to the format YYYY-MM-DD.
    - While telling dates to the user , tell them as 1st January , 2024 etc.
    - Response must be well formatted and in bullets and bolds.
    - Show every available flight to user with it's ID.
    - Once you have all the parameters , execute the tool without any further questions or conversation with user.
    - Log the API call parameters and response.


    """
    
    # Default to 'economy' if TravelClass is not specified
    TravelClass = TravelClass.lower() if TravelClass else 'economy'
    
    # Prepare the API endpoint
    url = 'https://bookmesky.com/air/api/search'
    
    # Set the headers with the authorization token
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {auth_token}'
    }
    
    # Handle trip type and traveling dates logic
    if trip_type == 'return' and arrive_date:
        dates = [depart_date, arrive_date]
    else:
        trip_type = 'one_way'
        dates = [depart_date]
        arrive_date = None  # For one-way trips, set arrive_date to None

    # List of airlines to call in parallel if ContentProvider is not provided
    airlines = ['airblue', 'airsial', 'sereneair', 'jazeera', 'flydubai', 'salamair', 'bookme-legacy', 'oneapi']

    # Function to make an API call for a specific airline
    def call_api(content_provider):
        data = {
            "Locations": [
                {"IATA": depart_from_city, "Type": "airport"},
                {"IATA": arrival_at_city, "Type": "airport"}
            ],
            "ContentProvider": content_provider,
            "Currency": "PKR",
            "TravelClass": TravelClass,
            "TripType": trip_type,
            "TravelingDates": dates,
            "Travelers": [
                {"Type": "adult", "Count": adult_count},
                {"Type": "child", "Count": child_count},
                {"Type": "infant", "Count": infant_count}
            ]
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raises an error if the response status is 4xx/5xx
            return response.json()

        except requests.exceptions.RequestException as e:
            # Handle any errors that occur during the API request
            return {"error": f"An error occurred for {content_provider}: {str(e)}"}

    results = []

    # If ContentProvider is not specified, call all airlines in parallel
    if not ContentProvider:
        with ThreadPoolExecutor() as executor:
            future_to_provider = {executor.submit(call_api, airline): airline for airline in airlines}

            for future in as_completed(future_to_provider):
                airline = future_to_provider[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": f"An error occurred for {airline}: {str(e)}"})
    else:
        # If ContentProvider is provided, just call the API for that specific airline
        results.append(call_api(ContentProvider))

    # Add the API response to the chat memory and return it
    formatted_responses = "\n".join([json.dumps(result, indent=2) for result in results])
    memory.chat_memory.add_message(AIMessage(content=f"API Responses:\n{formatted_responses}"))
    
    return results



tools =[
    get_auth_token, 
    get_airlines , 
    search_flights,
    today_date,
    staticTool
]



role =""" 
          1)You are BookieBee, an agent-based chatbot for Bookme.pk, designed to assist users with a variety of queries related to services offered by the company. Bookme.pk provides services including bus ticket booking, cinema ticketing, airline ticket reservations, and event ticket sales. Use the available tools to provide accurate, up-to-date, and helpful 
            responses regarding these services, ensuring a seamless customer experience.
            Even if the query isn't regrading the services , try your best to answer it using "StaticTool".
          2)Try to figure out inputs/arguments for the tools you need to use from the user's given query and if they are not found in the query you must ask them from user.
          3)Use current date and time as a reference to identify dates/times from user's queries. To get sure about current date/time use the provided tool "today_date".
          4)Always introduce yourself as BookieBee, making the conversation engaging and friendly.
          5)Use decent and professional emojis where appropriate to make interactions more appealing and light-hearted.
          6)Must not introduce yourself with every response.
          7)Must deal them as customers throughout the conversation.
          8)Response must be well formatted and in bullets and bolds.
      """

#Setting up buffer for LLM to have a look at.
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
    max_execution_time=4000000 , 
    handle_parsing_errors=True
)







