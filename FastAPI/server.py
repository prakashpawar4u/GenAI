from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

app = FastAPI()

# In-memory list to store items
items = []

class Item(BaseModel):
    id: int
    name: str
    description: str = None
    price: float

# GET: Retrieve all items
@app.get("/items/")
def get_items():
    return items

# GET: Retrieve an item by ID
@app.get("/items/{item_id}")
def get_item(item_id: int):
    for item in items:
        if item["id"] == item_id:
            return item
        raise HTTPException(status_code=404, detail="Item not found")

# POST: Add a new item
@app.post("/items/")
def create_item(item: Item):
    for existing_item in items:
        if existing_item["id"] == item.id:
            raise HTTPException(status_code=400, detail="Item with this ID already exists")
    items.append(item.dict())
    return item

# PUT: Update an existing item by ID
@app.put("/items/{item_id}")
def update_item(item_id: int, updated_item: Item):
    for index, item in enumerate(items):
        if item['id']==item_id:
            items[index]=updated_item.dict()
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")

# DELETE: Remove an item by ID
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    for index, item in enumerate(items):
        if item['id']==item_id:
            items.pop(index)
            return {"message":"Item deleted successfully"}
        
    raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)