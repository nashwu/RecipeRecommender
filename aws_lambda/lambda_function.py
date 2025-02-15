import json
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError

# initialize dynamodb
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("table_name")

# handles requests for storing and retrieving user preferences and chat history
def lambda_handler(event, context):

    try:
        body = json.loads(event["body"])
        action = body.get("action")

        if action == "store_preferences":
            response = store_preferences(body)
        elif action == "get_user_preferences":
            response = get_user_preferences(body)
        elif action == "store_chat_history":
            response = store_chat_history(body)
        elif action == "get_chat_history":
            response = get_chat_history(body)
        else:
            response = {"statusCode": 400, "body": json.dumps({"error": "invalid action"})}

    except json.JSONDecodeError:
        response = {"statusCode": 400, "body": json.dumps({"error": "invalid json format"})}
    except Exception as e:
        response = {"statusCode": 500, "body": json.dumps({"error": str(e)})}


    # return cors headers in every response
    response["headers"] = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "OPTIONS, GET, POST",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    return response



# stores user preferences in dynamodb
def store_preferences(data):

    user_id = data.get("user_id", "").strip()
    preferences = data.get("preferences", {})

    if not user_id:
        return {"statusCode": 400, "body": json.dumps({"error": "missing user_id"})}

    try:
        table.put_item(Item={"UserId": user_id, "preferences": preferences})
        return {"statusCode": 200, "body": json.dumps({"message": "preferences stored successfully"})}
    
    except (BotoCoreError, NoCredentialsError) as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}



# retrieves user preferences from dynamodb, creates default if not found
def get_user_preferences(data):

    user_id = data.get("user_id", "").strip()

    if not user_id:
        return {"statusCode": 400, "body": json.dumps({"error": "missing user_id"})}

    try:
        response = table.get_item(Key={"UserId": user_id})

        if "Item" in response:
            return {"statusCode": 200, "body": json.dumps(response["Item"].get("preferences", {}))}

        # create default preferences if user does not exist
        default_preferences = {
            "spicy": False,
            "vegan": False,
            "gluten_free": False,
            "favorite_cuisine": "none",
        }
        
        table.put_item(Item={"UserId": user_id, "preferences": default_preferences})

        return {"statusCode": 200, "body": json.dumps(default_preferences)}

    except (BotoCoreError, NoCredentialsError) as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}



# adds a chat entry to the user's chat history
def store_chat_history(data):
    user_id = data.get("user_id", "").strip()
    chat_entry = data.get("chat_entry", {})

    if not user_id or not chat_entry:
        return {"statusCode": 400, "body": json.dumps({"error": "missing user_id or chat_entry"})}

    try:
        response = table.get_item(Key={"UserId": user_id})
        chat_history = response.get("Item", {}).get("chat_history", [])

        # append new chat entry
        chat_history.append(chat_entry)

        # update chat history in dynamodb
        table.update_item(
            Key={"UserId": user_id},
            UpdateExpression="SET chat_history = :chat_history",
            ExpressionAttributeValues={":chat_history": chat_history}
        )


        return {"statusCode": 200, "body": json.dumps({"message": "chat history updated successfully"})}

    except (BotoCoreError, NoCredentialsError) as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}



# retrieves the chat history for a given user
def get_chat_history(data):
    user_id = data.get("user_id", "").strip()

    if not user_id:
        return {"statusCode": 400, "body": json.dumps({"error": "missing user_id"})}

    try:
        response = table.get_item(Key={"UserId": user_id})

        if "Item" in response and "chat_history" in response["Item"]:
            return {"statusCode": 200, "body": json.dumps(response["Item"]["chat_history"])}


        return {"statusCode": 200, "body": json.dumps([])}


    except (BotoCoreError, NoCredentialsError) as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
