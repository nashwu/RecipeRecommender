from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import torch
import logging
import requests
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ultralytics import YOLO
import cv2
import numpy as np
import os

# basic logging setup
logging.basicConfig(level=logging.DEBUG)

# load llm model
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# load nlp models
nlp = spacy.load("en_core_web_sm")  # replace with FoodBaseBERT-NER model
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# aws lambda api gateway url replace w/ urs
API_GATEWAY_URL = ""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cv_model_path = os.path.join(BASE_DIR, "api", "best.pt")
cv_model = YOLO(cv_model_path)


# for calling lambda functions
def call_lambda(action, data):
    try:
        response = requests.post(
            API_GATEWAY_URL,
            headers={"Content-Type": "application/json"},
            json={"action": action, **data},
            timeout=5
        )


        response.raise_for_status()
        logging.debug(f"lambda response: {response.text}")

        return response.json()
    
    except requests.RequestException as e:
        logging.error(f"error calling lambda: {e}")
        return {"error": str(e), "details": response.text if 'response' in locals() else "no response"}


# generate recipe
@csrf_exempt
@require_http_methods(["POST", "OPTIONS"])
def generate_recipe(request):

    try:
        data = json.loads(request.body)
        user_input = data.get("prompt", "").strip()
        user_id = data.get("user_id", "").strip()

        if not user_input or not user_id:
            return JsonResponse({"error": "missing user_id or prompt"}, status=400)

        # get user preferences from lambda
        preferences_response = call_lambda("get_user_preferences", {"user_id": user_id})
        preferences = json.loads(preferences_response["body"]) if "body" in preferences_response else {}

        # format the prompt
        formatted_prompt = construct_prompt(user_input, preferences)

        # generate response
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda")
        output = llm_model.generate(input_ids, max_length=2000, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(output[0], skip_special_tokens=True).split("### Response:\n")[-1].strip()

        # store chat history
        store_chat_response = call_lambda("store_chat_history", {
            "user_id": user_id,
            "chat_entry": {"prompt": user_input, "recipe": response_text}
        })

        if "error" in store_chat_response:
            logging.error(f"error storing chat history: {store_chat_response['error']}")

        return JsonResponse({"recipe": response_text}, json_dumps_params={'indent': 2})
    
    except Exception as e:
        logging.error(f"error generating recipe: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# stores user preferences w/ aws lambda
@csrf_exempt
@require_http_methods(["POST"])
def store_preferences(request):
    try:
        data = json.loads(request.body)
        user_id = data.get("user_id", "").strip()
        preferences = data.get("preferences", {})

        if not user_id:
            return JsonResponse({"error": "missing user_id"}, status=400)

        response = call_lambda("store_preferences", {"user_id": user_id, "preferences": preferences})
        return JsonResponse(response)
    
    except Exception as e:
        logging.error(f"unexpected error storing preferences: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# gets chat hist w/ aws lambda
@csrf_exempt
@require_http_methods(["POST"])
def get_chat_history(request):
    try:
        data = json.loads(request.body)
        user_id = data.get("user_id", "").strip()

        if not user_id:
            return JsonResponse({"error": "missing user_id"}, status=400)

        response = call_lambda("get_chat_history", {"user_id": user_id})
        return JsonResponse(response)
    
    except Exception as e:
        logging.error(f"unexpected error retrieving chat history: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# anaylze + store
@csrf_exempt
@require_http_methods(["POST"])
def analyze_and_store_preferences(request):
    try:
        data = json.loads(request.body)
        user_id = data.get("user_id", "").strip()
        chat_message = data.get("message", "").strip()

        if not user_id or not chat_message:
            return JsonResponse({"error": "missing user_id or message"}, status=400)

        # extract ingredients and analyze sentiment
        ingredients = extract_ingredients(chat_message)
        preferences = analyze_sentiment(ingredients, chat_message)

        # store preferences in dynamodb
        response = call_lambda("store_preferences", {"user_id": user_id, "preferences": preferences})
        return JsonResponse(response)
    
    except Exception as e:
        logging.error(f"error processing preferences: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# detect ingredients w/ yolo
@csrf_exempt
def detect_ingredients(request):

    if request.method == "POST" and request.FILES.get("image"):

        image = request.FILES["image"].read()
        np_img = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        results = cv_model(img)
        detected_ingredients = set()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = cv_model.names[class_id]
                detected_ingredients.add(class_name)

        return JsonResponse({"ingredients": list(detected_ingredients)})

    return JsonResponse({"error": "invalid request"}, status=400)


# finding food ingredients in user text
def extract_ingredients(text):
    doc = nlp(text)
    ingredients = [ent.text.lower() for ent in doc.ents if ent.label_ == "FOOD"]
    return list(set(ingredients))



# anaylze sent in user text
def analyze_sentiment(ingredients, text):

    preferences = {"positive_ingredients": [], "negative_ingredients": []}

    for ingredient in ingredients:
        if ingredient in text:
            sentiment_result = sentiment_analyzer(ingredient + " " + text)[0]
            score = sentiment_result["label"]
            
            if "5" in score or "4" in score:
                preferences["positive_ingredients"].append(ingredient)
            elif "1" in score or "2" in score:
                preferences["negative_ingredients"].append(ingredient)

    return preferences



# construct prompt for llm model
def construct_prompt(user_input, preferences):

    preference_text = "\n".join([f"**{key.replace('_', ' ').title()}**: {str(value).capitalize()}" for key, value in preferences.items()])
    
    prompt = (
        "### Instruction:\n"
        f"Generate a detailed and creative recipe based on the following user request in under 500 words.\n"
        f"**Request**: {user_input}\n"
    )
    
    if preferences:
        prompt += f"**User Preferences**:\n{preference_text}\n"
    
    prompt += "### Response:\n"
    return prompt
