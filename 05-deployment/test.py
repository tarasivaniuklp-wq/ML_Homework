import requests

url = 'https://churn-prediction-agent.onrender.com/predict'

customer = {
    "gender": "female",
    "seniorcitizen": 0,
}

response = requests.post(url, json=customer).json()
print(response)

# if response['churn'] == True:
#     print('sending promo email to %s' % customer_id)
# else:
#     print('not sending promo email to %s' % customer_id)