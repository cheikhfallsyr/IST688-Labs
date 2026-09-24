import json
import requests
import streamlit as st
from openai import OpenAI


# Part A
def get_current_weather(location):
    location = (location or "").strip() or "Syracuse, NY"

    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception(
            f"wttr.in error: status {response.status_code}"
        )

    try:
        data = response.json()
    except ValueError:
        raise Exception(
            f"Could not find a location named {location}"
        )

    current = data["current_condition"][0]
    today = data["weather"][0]
    nearest_area = data["nearest_area"][0]

    matched_location = ", ".join(
        value
        for value in [
            nearest_area["areaName"][0]["value"],
            nearest_area["region"][0]["value"],
            nearest_area["country"][0]["value"],
        ]
        if value
    )
    hourly_forecast = []

    for hour in today["hourly"]:
        hour_number = int(hour["time"])
        time_label = (
            f"{hour_number // 100:02d}:"
            f"{hour_number % 100:02d}"
        )

        hourly_forecast.append(
            {
                "time": time_label,
                "temperature_f": float(hour["tempF"]),
                "feels_like_f": float(hour["FeelsLikeF"]),
                "description": hour["weatherDesc"][0]["value"],
                "chance_of_rain_percent": int(hour["chanceofrain"]),
            }
        )

    return {
        "requested_location": location,
        "matched_location": matched_location,
        "current_temperature_f": float(current["temp_F"]),
        "current_feels_like_f": float(current["FeelsLikeF"]),
        "current_description": current["weatherDesc"][0]["value"],
        "humidity_percent": int(current["humidity"]),
        "wind_speed_mph": float(current["windspeedMiles"]),
        "wind_direction": current["winddir16Point"],
        "precipitation_inches": float(current["precipInches"]),
        "uv_index": int(current["uvIndex"]),
        "today_minimum_f": float(today["mintempF"]),
        "today_maximum_f": float(today["maxtempF"]),
        "sunrise": today["astronomy"][0]["sunrise"],
        "sunset": today["astronomy"][0]["sunset"],
        "hourly_forecast": hourly_forecast,
    }
# Part B
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current weather and today's forecast for a location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "A city, zip code, airport code, or landmark. "
                        "Use Syracuse, NY if no location is provided."
                    ),
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)


# Streamlit interface.
st.title("Lab 5: What to Wear Bot")

st.write(
    "Enter a city to receive clothing and outdoor activity "
    "suggestions based on today's weather."
)

location_input = st.text_input(
    "City",
    placeholder="For example: Syracuse, NY",
)


if st.button("Get suggestions"):
    location = location_input.strip() or "Syracuse, NY"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a weather-based clothing and activity assistant. "
                "Use the get_current_weather tool when weather information "
                "is needed. If no location is provided, use Syracuse, NY. "
                "After weather data is returned, recommend appropriate "
                "clothes for today and outdoor activities suitable for "
                "the conditions. Consider the temperature, feels-like "
                "temperature, wind, rain, UV index, and changes throughout "
                "the day. Identify the location returned by the weather "
                "service and do not invent weather information."
            ),
        },
        {
            "role": "user",
            "content": (
                f"What should I wear today in {location}, and what "
                "outdoor activities would be appropriate?"
            ),
        },
    ]

    try:
        with st.spinner("Checking the weather..."):

            # First API call
            first_response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                tools=[weather_tool],
                tool_choice="auto",
            )

            assistant_message = first_response.choices[0].message

            if assistant_message.tool_calls:
                messages.append(assistant_message)

                
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "get_current_weather":
                        arguments = json.loads(
                            tool_call.function.arguments
                        )

                        tool_location = (
                            arguments.get("location") or "Syracuse, NY"
                        )

                        weather_data = get_current_weather(
                            tool_location
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(weather_data),
                            }
                        )

                # Second API call
                final_response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=messages,
                )

                advice = final_response.choices[0].message.content

            else:
                advice = assistant_message.content

        st.subheader("Today's suggestions")
        st.write(advice)

    except Exception as error:
        st.error(str(error))