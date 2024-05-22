def suggest_lane(cars_detected):
    # Assuming cars_detected is a dictionary where keys represent lanes and values represent the number of cars detected in each lane
    # Example: cars_detected = {'Lane 1': 3, 'Lane 2': 2, 'Lane 3': 4}
    
    # Sort lanes by the number of cars detected
    sorted_lanes = sorted(cars_detected.items(), key=lambda x: x[1])

    # Choose the lane with the fewest cars
    suggested_lane = sorted_lanes[0][0]

    return suggested_lane

# Example usage:
if __name__ == "__main__":
    # Example dictionary representing the number of cars detected in each lane
    cars_detected = {'Lane 1': 3, 'Lane 2': 2, 'Lane 3': 4}

    # Get the suggested lane
    suggested_lane = suggest_lane(cars_detected)

    print("Suggested lane:", suggested_lane)
