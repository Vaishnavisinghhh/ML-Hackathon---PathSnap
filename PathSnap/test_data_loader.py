
from src.data_loader import load_trip, load_all_trips

# Test one training trip
trip = load_trip("data/Student folder/train/00000001")

print("========== SINGLE TRIP TEST ==========")
print("Trip ID:", trip["trip_id"])
print("GPS observations:", len(trip["track"]))
print("Road nodes:", len(trip["nodes"]))
print("Road arcs:", len(trip["arcs"]))
print("Route entries:", len(trip["route"]))

print("\nFirst GPS observation:", trip["track"][0])
print("First road node:", trip["nodes"][0])
print("First road arc:", trip["arcs"][0])
print("First route entry:", trip["route"][0])

# Test loading all training trips
trips = load_all_trips("data/Student folder/train")

print("\n========== ALL TRAINING TRIPS ==========")
print("Number of trips loaded:", len(trips))

print("\nAll trips loaded successfully!")
