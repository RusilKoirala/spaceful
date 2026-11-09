import os

BRAIN_SIZES = [8, 10, 8, 4]  
POPULATION = 200            
BREEDING_POOL = 50           
MUTATION = 0.1               
CRASH_PENALTY = 0.5
FINISH_BONUS = 2            
STOP_WHEN_FINISHED = False   
BRAIN_FILE = os.path.join(os.path.dirname(__file__), "breeding_pool.pkl")
