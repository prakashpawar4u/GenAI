def simplify_directions(directions):
    # Define opposites of each direction
    opposites = {"North": "South", "South": "North", "East": "West", "West": "East"}
    
    # Initialize a stack to store the simplified path
    stack = []

    # Iterate over each direction
    for direction in directions:        
        # If the stack is not empty and the current direction cancels the last one, pop the stack
        if stack and stack[-1] == opposites.get(direction):
            stack.pop()  # Remove the opposite direction
        else:
            # Otherwise, add the current direction to the stack
            stack.append(direction)
    
    return stack

# Test input
#directions = ["North", "North", "East", "South", "West", "West"]
directions = ["North", "South", "East", "South", "West", "East"]
output = simplify_directions(directions)
print("Final Output:", output)
