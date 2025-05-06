import redis
import pickle

r = redis.Redis(host="localhost", port=6379, db=0)
def save_cache(key, value):
    r.set(key, pickle.dumps(value))

def load_cache(key):
    data = r.get(key)
    return pickle.loads(data) if data else None

def has_cache(key):
    return r.exists(key)
