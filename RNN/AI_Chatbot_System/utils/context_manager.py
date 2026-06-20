conversation_memory = []

def add_message(msg):
    conversation_memory.append(msg)

def get_context():
    return conversation_memory[-5:]