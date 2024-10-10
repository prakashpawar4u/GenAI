import requests

BASE_URL = "http://127.0.0.1:8000/items/"

# POST: Add a new item
def create_item(item):
    response = requests.post(BASE_URL, json=item)
    print("POST Response:", response.json())

# GET: Retrieve all items
def get_items():
    response = requests.get(BASE_URL)
    print("GET Response (all items):", response.json())

# GET: Retrieve a specific item by ID
def get_item(item_id):
    response = requests.get(f"{BASE_URL}{item_id}")
    if response.status_code == 200:
        print("GET Response (item):", response.json())
    else:
        print("GET Error:", response.json())

# PUT: Update an existing item
def update_item(item_id, updated_item):
    response = requests.put(f"{BASE_URL}{item_id}", json=updated_item)
    if response.status_code == 200:
        print("PUT Response:", response.json())
    else:
        print("PUT Error:", response.json())

# DELETE: Remove an item by ID
def delete_item(item_id):
    response = requests.delete(f"{BASE_URL}{item_id}")
    if response.status_code == 200:
        print("DELETE Response:", response.json())
    else:
        print("DELETE Error:", response.json())       

# Example usage of the API:
if __name__ == "__main__":
    # Creating items
    item1 = {"id": 1, "name": "Item One", "description": "This is item one", "price": 10.99}
    item2 = {"id": 2, "name": "Item Two", "description": "This is item two", "price": 20.99}
    
    create_item(item1)
    create_item(item2)

    # Get all items
    get_items()

    # Get a single item by ID
    get_item(1)