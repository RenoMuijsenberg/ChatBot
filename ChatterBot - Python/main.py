# ChatBot For Semester 6 using ChatterBot 2

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot("bot")


def train():
    trainer = ChatterBotCorpusTrainer(chatbot)

    trainer.train(
        "data/english/"
    )


def start():
    exit_conditions = (":q", "quit", "exit")
    while True:
        query = input("> ")
        if query in exit_conditions:
            break
        else:
            print(f"🪴 {chatbot.get_response(query)}")


if __name__ == "__main__":
    train()
    start()
